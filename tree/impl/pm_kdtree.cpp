#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include <common/util/utils.h>
#include <tree/helper.h>
#include <tree/pm_kdtree.h>
#include <tree/kernel.h>

namespace pmkd {

	// PMKDTree implementation
	PMKDTree::PMKDTree() { init(); }

	PMKDTree::PMKDTree(const PMKD_Config& config) {
		this->config = config;
		this->config.expandFactor = fmin(this->config.expandFactor, 1.2f);

		init();
	}

	void PMKDTree::init() {
		ptNum = 0;
		bufferPool = std::make_unique<BufferPool>();
		// set config
		globalBoundary = config.globalBoundary;
	}

	PMKDTree::~PMKDTree() { destroy(); }

	void PMKDTree::destroy() {
		// release stored points
		// release leaves and interiors
		ptNum = 0;
	}

	void PMKDTree::expandStorage(size_t newCapacity) {
		leaves.reserve(newCapacity);

		// note: allocation of interiors can be async
		interiors.reserve(newCapacity);
	}

	void PMKDTree::buildStatic() {
		if (ptNum == 0) {
			return;
		}
		// get scene boundary
		//for (size_t i = 0; i < ptNum; ++i) { sceneBoundary.merge(pts[i]); }
		sceneBoundary = reduce<AABB>(pts, MergeOp());
		globalBoundary.merge(sceneBoundary);

		// init leaves
		leaves.morton.resize(ptNum);
		auto primIdx = parlay::to_sequence(parlay::iota<int>(ptNum));
		//leaves.segOffset.resize(ptNum);

		// init interiors
		// note: allocation of interiors can be async
		interiors.rangeL.resize(ptNum - 1);
		interiors.rangeR.resize(ptNum - 1);
		interiors.splitDim.resize(ptNum - 1);
		interiors.splitVal.resize(ptNum - 1);
		interiors.parentSplitDim.resize(ptNum - 1);
		interiors.parentSplitVal.resize(ptNum - 1);

		// calculate morton code
		parlay::parallel_for(0, ptNum,
			[&](size_t i) { BuildKernel::calcMortonCodes(i, ptNum, pts.data(), &globalBoundary, leaves.morton.data()); }
		);

		// reorder leaves using morton code
		// note: there are multiple sorting algorithms to choose from
		parlay::integer_sort_inplace(primIdx, [&](const auto& idx) {return leaves.morton[idx].code;});
		auto mortonSorted = parlay::tabulate(ptNum, [&](int i) {return leaves.morton[primIdx[i]];});
		leaves.morton = std::move(mortonSorted);
		auto ptsSorted = parlay::tabulate(ptNum, [&](int i) {return pts[primIdx[i]];});
		pts = std::move(ptsSorted);

		// calculate metrics
		auto metrics = bufferPool->acquire<int>(ptNum - 1);

		parlay::parallel_for(0, ptNum - 1,
			[&](size_t i) {
				BuildKernel::calcBuildMetrics(
					i, ptNum - 1, globalBoundary, leaves.morton.data(), metrics.data(),
					interiors.splitDim.data(), interiors.splitVal.data());
			}
		);

		auto visitCount = vector<AtomicCount>(ptNum - 1);
		auto innerBuf = bufferPool->acquire<int>(ptNum - 1);
		auto leafBuf = bufferPool->acquire<int>(ptNum);
		BuildAid aid{ metrics.data(),visitCount.data(),innerBuf.data(),leafBuf.data() };
		// build interior nodes
		if (config.optimize) {
			int* range[2] = { interiors.rangeL.data(),interiors.rangeR.data() };
			parlay::parallel_for(0, ptNum,
				[&](size_t i) {
					BuildKernel::buildInteriors_opt(
						i, ptNum, leaves.getRawRepr(),
						range, interiors.splitDim.data(), interiors.splitVal.data(),
						interiors.parentSplitDim.data(), interiors.parentSplitVal.data(),
						aid);
				}
			);
		}
		else {
			parlay::parallel_for(0, ptNum,
				[&](size_t i) {
					BuildKernel::buildInteriors(
						i, ptNum, leaves.getRawRepr(), interiors.getRawRepr(), aid);
				}
			);
		}

		// calculate new indices for interiors
		auto& segLen = leafBuf;
		leaves.segOffset = parlay::scan(segLen).first;

		auto& mapidx = metrics;
		parlay::parallel_for(0, ptNum - 1,
			[&](size_t i) {
				BuildKernel::calcInteriorNewIdx(
					i, ptNum-1, leaves.getRawRepr(), interiors.getRawRepr(), aid.segLen,aid.leftLeafCount, mapidx.data());
			}
		);

		// reorder interiors
		auto& rangeL = innerBuf;
		auto& rangeR = leafBuf;
		rangeR.resize(ptNum - 1);
		auto splitDim = bufferPool->acquire<int>(ptNum - 1);
		auto splitVal = bufferPool->acquire<mfloat>(ptNum - 1);
		parlay::parallel_for(0, ptNum - 1,
			[&](size_t i) {
				BuildKernel::reorderInteriors_step1(
					i, ptNum - 1, interiors.getRawRepr(), mapidx.data(),
					rangeL.data(), rangeR.data(),
					splitDim.data(), splitVal.data());
			}
		);
		bufferPool->release<int>(std::move(interiors.rangeL));
		bufferPool->release<int>(std::move(interiors.rangeR));
		bufferPool->release<int>(std::move(interiors.splitDim));
		bufferPool->release<mfloat>(std::move(interiors.splitVal));

		interiors.rangeL = std::move(rangeL);
		interiors.rangeR = std::move(rangeR);
		interiors.splitDim = std::move(splitDim);
		interiors.splitVal = std::move(splitVal);

		auto parentSplitDim = bufferPool->acquire<int>(ptNum - 1);
		auto parentSplitVal = bufferPool->acquire<mfloat>(ptNum - 1);
		parlay::parallel_for(0, ptNum - 1,
			[&](size_t i) {
				BuildKernel::reorderInteriors_step2(
					i, ptNum - 1, interiors.getRawRepr(), mapidx.data(),
					parentSplitDim.data(), parentSplitVal.data());
			}
		);
		bufferPool->release<int>(std::move(interiors.parentSplitDim));
		bufferPool->release<mfloat>(std::move(interiors.parentSplitVal));
		bufferPool->release<int>(std::move(mapidx));

		interiors.parentSplitDim = std::move(parentSplitDim);
		interiors.parentSplitVal = std::move(parentSplitVal);
	}

	void PMKDTree::buildIncrement(const vector<vec3f>& ptsAdd) {
		size_t sizeInc = ptsAdd.size();
		auto binIdx = bufferPool->acquire<int>(sizeInc);
		auto primIdx = bufferPool->acquire<int>(sizeInc);
		auto morton = bufferPool->acquire<MortonType>(sizeInc);
		auto mortonSorted = bufferPool->acquire<MortonType>(sizeInc);
		auto ptsAddSorted = bufferPool->acquire<vec3f>(sizeInc);

		// get scene boundary
		sceneBoundary.merge(reduce<AABB>(ptsAdd, MergeOp()));
		assert(globalBoundary.include(sceneBoundary));

		// calculate morton code
		parlay::parallel_for(0, sizeInc,
			[&](size_t i) { BuildKernel::calcMortonCodes(i, sizeInc, ptsAdd.data(), &globalBoundary, morton.data()); }
		);
		// assign primIdx
		parlay::parallel_for(0, sizeInc,
            [&](size_t i) { primIdx[i] = i; }
		);
		// sort primIdx
		parlay::integer_sort_inplace(
			primIdx,
			[&](const auto& idx) {return morton[idx].code;});

		// sort ptsAdd
		parlay::parallel_for(0, sizeInc,
			[&](size_t i) { ptsAddSorted[i] = ptsAdd[primIdx[i]]; }
		);
		// sort morton
		parlay::parallel_for(0, sizeInc,
			[&](size_t i) { mortonSorted[i] = morton[primIdx[i]]; }
		);

		// find leaf bin
		int maxBin = -1;
		parlay::parallel_for(0, sizeInc,
			[&](size_t i) {
				UpdateKernel::findLeafBin(
					i, sizeInc, ptsAddSorted.data(), primSize(),
					interiors.getRawRepr(), leaves.getRawRepr(), sceneBoundary, binIdx.data());
			});
		maxBin = parlay::reduce(binIdx, parlay::maximum<int>());

		// reset primIdx
		parlay::parallel_for(0, sizeInc, [&](uint32_t i) {return primIdx[i] = ptNum + i;});

		// calc metrics
		// build interiors

		bufferPool->release<int>(std::move(binIdx));
		bufferPool->release<int>(std::move(primIdx));
		bufferPool->release<MortonType>(std::move(morton));
		bufferPool->release<MortonType>(std::move(mortonSorted));
		bufferPool->release<vec3f>(std::move(ptsAddSorted));
	}

	
	PMKD_PrintInfo PMKDTree::print(bool verbose) const {
		PMKD_PrintInfo info;
		info.leafNum = ptNum;
		if (ptNum == 0) return info;

		using PNode = std::pair<int, int>;

		auto left = [&](const PNode& pnode) ->PNode {
			auto& [leafBinIdx, node] = pnode;
			if (node >= ptNum) return { -1,-1 };  // leaf has no child
			int R = leaves.segOffset[leafBinIdx + 1];
			int next = node < R - 1 ? node + 1 : interiors.rangeL[node] + ptNum;
			return { leafBinIdx,next };
		};
		
		auto right = [&](const PNode& pnode) ->PNode {
			auto& [leafBinIdx, node] = pnode;
			if (node >= ptNum) return { -1,-1 };  // leaf has no child

			int R = leaves.segOffset[leafBinIdx + 1];
			int nextBin = node == R - 1 ? leafBinIdx + 1 : interiors.rangeR[node + 1] + 1;
			int nextIdx = leaves.segOffset[nextBin];

			if (nextBin == ptNum - 1) {
				assert(nextBin == nextIdx);
				nextIdx += ptNum;
			}
			else if (nextIdx == leaves.segOffset[nextBin + 1]) nextIdx = nextBin + ptNum;
			
			return { nextBin,nextIdx };
		};

		auto isInvalid = [](const PNode& node) {return node.first == -1;};

		std::vector<PNode> stk;
		// inorder traversal
		PNode p{ 0,0 };
		while (!isInvalid(p) || !stk.empty()) {
			if (!isInvalid(p)) {
				stk.push_back(p);
				p = left(p);
			}
			else {
				p = stk[stk.size() - 1];
				stk.pop_back();
				info.inorderTraversal.push_back(p.second);
				p = right(p);
			}
		}

		// preorder traversal
		int L = 0, R = 0;
		int interiorIdx = 0;
		for (int begin = 0; begin < ptNum; begin++) {
			L = leaves.segOffset[begin];
			R = begin == ptNum - 1 ? L : leaves.segOffset[begin + 1];
			// interiors
			for (interiorIdx = L; interiorIdx < R; interiorIdx++) {
				info.preorderTraversal.push_back(interiorIdx);
			}
			// leaf
			info.preorderTraversal.push_back(begin + ptNum);
		}

		// get leaf mortons
		info.leafMortons.assign(leaves.morton.begin(), leaves.morton.end());

		// get morton metrics
		info.metrics.resize(ptNum - 1);
		parlay::parallel_for(0, ptNum - 1,
			[&](size_t i) {
				info.metrics[i] = MortonType::calcMetric(info.leafMortons[i], info.leafMortons[i + 1]);
			}
		);

		if (verbose) {
			// get primIdx
			info.leafPoints.resize(ptNum);
			//parlay::parallel_for(0, ptNum, [&](size_t i) { info.leafPoints[i] = pts[leaves.primIdx[i]];});
			info.leafPoints.assign(pts.begin(), pts.end());
			info.splitDim.assign(interiors.splitDim.begin(), interiors.splitDim.end());
			info.splitVal.assign(interiors.splitVal.begin(), interiors.splitVal.end());
		}
		return info;
	}

	QueryResponses PMKDTree::query(const vector<Query>& queries) const {
		if (queries.empty()) return QueryResponses(0);

		size_t nq = queries.size();
		QueryResponses responses(nq);

		// sort queries to improve cache friendlyness
		if (config.optimize) {
			vector<MortonType> morton(nq);

			parlay::parallel_for(0, nq,
				[&](size_t i) { BuildKernel::calcMortonCodes(i, nq, queries.data(), &globalBoundary, morton.data()); }
			);
			// note: there are multiple sorting algorithms to choose from
			parlay::integer_sort_inplace(
				responses.queryIdx, [&](const auto& idx) {return morton[idx].code;});
			auto queriesSorted = parlay::tabulate(nq, [&](int i) {return queries[responses.queryIdx[i]];});

			parlay::parallel_for(0, nq,
				[&](size_t i) {
					SearchKernel::searchPoints(
						i, nq, queriesSorted.data(), pts.data(), ptNum,
						interiors.getRawRepr(), leaves.getRawRepr(), sceneBoundary, responses.exist.data());
				}
			);
		}
		else {
			parlay::parallel_for(0, nq,
				[&](size_t i) {
					SearchKernel::searchPoints(
						i, nq, queries.data(), pts.data(), ptNum,
						interiors.getRawRepr(), leaves.getRawRepr(), sceneBoundary, responses.exist.data());
				}
			);
		}

		return responses;
	}

	RangeQueryResponses PMKDTree::query(const vector<RangeQuery>& queries) const {
		if (queries.empty()) return RangeQueryResponses(0);
		
		size_t nq = queries.size();
		RangeQueryResponses responses(nq);

		// note: sort queries as an optimization
		if (config.optimize) {}

		parlay::parallel_for(0, nq,
			[&](size_t i) {
				SearchKernel::searchRanges(
					i, nq, queries.data(), pts.data(), ptNum,
					interiors.getRawRepr(), leaves.getRawRepr(),
					sceneBoundary, responses.getRawRepr());
			}
		);
		return responses;
	}

	void PMKDTree::insert(const vector<vec3f>& ptsAdd) {
		if (ptsAdd.empty()) return;
		buildIncrement(ptsAdd);
	}

	void PMKDTree::remove(const vector<vec3f>& ptsRemove) {}
}