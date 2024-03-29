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
		init();
	}

	void PMKDTree::init() {
		isStatic = false;
		nTotalDInserted = 0;
		nTotalRemoved = 0;

		nodeMgr = std::make_unique<NodeMgr>();
		bufferPool = std::make_unique<BufferPool>();
		// set config
		globalBoundary = config.globalBoundary;
	}

	PMKDTree::~PMKDTree() { destroy(); }

	void PMKDTree::destroy() {
		// release stored points
		// release leaves and interiors
		nodeMgr->clear();
		//sceneBoundary.reset();
		globalBoundary = config.globalBoundary;
		isStatic = false;
		nTotalDInserted = 0;
		nTotalRemoved = 0;
	}

	void PMKDTree::sortPts(const vector<vec3f>& pts, vector<vec3f>& ptsSorted) const {
		size_t nPts = pts.size();

		auto primIdx = bufferPool->acquire<int>(nPts);
		auto morton = bufferPool->acquire<MortonType>(nPts);

		parlay::parallel_for(0, nPts,
			[&](size_t i) { primIdx[i] = i; }
		);

		parlay::parallel_for(0, nPts,
			[&](size_t i) { BuildKernel::calcMortonCodes(i, nPts, pts.data(), &globalBoundary, morton.data()); }
		);
		// note: there are multiple sorting algorithms to choose from
		parlay::integer_sort_inplace(
			primIdx, [&](const auto& idx) {return morton[idx].code;});

		parlay::parallel_for(0, nPts, [&](size_t i) {ptsSorted[i] = pts[primIdx[i]]; });

		bufferPool->release(std::move(primIdx));
		bufferPool->release(std::move(morton));
	}

	void PMKDTree::sortPts(const vector<vec3f>& pts, vector<vec3f>& ptsSorted, vector<int>& primIdxInited) const {
		size_t nPts = pts.size();

		auto morton = bufferPool->acquire<MortonType>(nPts);

		parlay::parallel_for(0, nPts,
			[&](size_t i) { BuildKernel::calcMortonCodes(i, nPts, pts.data(), &globalBoundary, morton.data()); }
		);
		// note: there are multiple sorting algorithms to choose from
		parlay::integer_sort_inplace(
			primIdxInited, [&](const auto& idx) {return morton[idx].code;});

		parlay::parallel_for(0, nPts, [&](size_t i) {ptsSorted[i] = pts[primIdxInited[i]]; });

		bufferPool->release(std::move(morton));
	}

	void PMKDTree::sortPts(const vector<vec3f>& pts, vector<vec3f>& ptsSorted, vector<int>& primIdx, vector<MortonType>& morton) const {
		size_t nPts = pts.size();

		parlay::parallel_for(0, nPts,
			[&](size_t i) { primIdx[i] = i; }
		);

		parlay::parallel_for(0, nPts,
			[&](size_t i) { BuildKernel::calcMortonCodes(i, nPts, pts.data(), &globalBoundary, morton.data()); }
		);
		// note: there are multiple sorting algorithms to choose from
		parlay::integer_sort_inplace(
			primIdx, [&](const auto& idx) {return morton[idx].code;});

		parlay::parallel_for(0, nPts, [&](size_t i) {ptsSorted[i] = pts[primIdx[i]]; });
	}

	void PMKDTree::buildStatic(const vector<vec3f>& pts) {
		size_t ptNum = pts.size();

		// note: can be async
		vector<vec3f> ptsSorted(ptNum);
		// init leaves
		Leaves leaves;
		leaves.resizePartial(ptNum);  // note: can be async
		//leaves.segOffset.resize(ptNum);

		// init interiors
		Interiors interiors;
		// note: allocation of interiors can be async
		interiors.resize(ptNum - 1);

		// get scene boundary
		//sceneBoundary = reduce<AABB>(pts, MergeOp());
		//globalBoundary.merge(sceneBoundary);

		auto primIdx = bufferPool->acquire<int>(ptNum);
		auto _morton = bufferPool->acquire<MortonType>(ptNum);

		sortPts(pts, ptsSorted, primIdx, _morton);
		parlay::parallel_for(0, ptNum, [&](size_t i) {leaves.morton[i] = _morton[primIdx[i]];});

		bufferPool->release(std::move(primIdx));
		bufferPool->release(std::move(_morton));

#ifdef ENABLE_MERKLE
		// calc leaf hash
		parlay::parallel_for(0, ptNum,
			[&](size_t i) {
				BuildKernel::calcLeafHash(i, ptNum, ptsSorted.data(), leaves.replacedBy.data(), leaves.hash.data());
			}
		);
#endif

		buildStatic_LeavesReady(leaves, interiors);
		nodeMgr->append(std::move(leaves), std::move(interiors), std::move(ptsSorted));
	}

	void PMKDTree::buildStatic_LeavesReady(Leaves& leaves, Interiors& interiors) {
		size_t ptNum = leaves.size();


		auto metrics = bufferPool->acquire<uint8_t>(ptNum - 1);
		auto mapidx = bufferPool->acquire<int>(ptNum - 1);
		auto splitDim = bufferPool->acquire<int>(ptNum - 1);
		auto splitVal = bufferPool->acquire<mfloat>(ptNum - 1);
		auto parent = bufferPool->acquire<int>(ptNum - 1);

		// build aid
		auto visitCount = vector<AtomicCount>(ptNum - 1);
		auto innerBuf = bufferPool->acquire<int>(ptNum - 1, 0);
		auto leafBuf = bufferPool->acquire<int>(ptNum, 0);

		// calculate metrics
		parlay::parallel_for(0, ptNum - 1,
			[&](size_t i) {
				BuildKernel::calcBuildMetrics(
					i, ptNum - 1, globalBoundary, leaves.morton.data(), metrics.data(),
					interiors.splitDim.data(), interiors.splitVal.data());
			}
		);

		BuildAid aid{ metrics.data(), visitCount.data(), innerBuf.data(),leafBuf.data() };

		// build interior nodes
		// note: the following optimization does not prove better on CPU
		// may be better on GPU
		// if (config.optimize) {
		// 	int* range[2] = { interiors.rangeL.data(),interiors.rangeR.data() };
		// 	parlay::parallel_for(0, ptNum,
		// 		[&](size_t i) {
		// 			BuildKernel::buildInteriors_opt(
		// 				i, ptNum, leaves.getRawRepr(),
		// 				range, interiors.splitDim.data(), interiors.splitVal.data(),
		// 				interiors.parentSplitDim.data(), interiors.parentSplitVal.data(),
		// 				aid);
		// 		}
		// 	);
		// }
		// else {
		parlay::parallel_for(0, ptNum,
			[&](size_t i) {
				BuildKernel::buildInteriors(
					i, ptNum, leaves.getRawRepr(), interiors.getRawRepr(), aid);
			}
		);
		//}
		bufferPool->release(std::move(metrics));

		// calculate new indices for interiors
		auto& segLen = leafBuf;
		leaves.segOffset = parlay::scan(segLen).first;

		//fmt::print("segOffset: {}\n", leaves.segOffset);

		parlay::parallel_for(0, ptNum - 1,
			[&](size_t i) {
				BuildKernel::calcInteriorNewIdx(
					i, ptNum - 1, leaves.getRawRepr(), interiors.getRawRepr(), aid.segLen, aid.leftLeafCount, mapidx.data());
			}
		);
		// reorder interiors
		auto& rangeL = innerBuf;
		auto& rangeR = leafBuf;
		rangeR.resize(ptNum - 1);

		parlay::parallel_for(0, ptNum - 1,
			[&](size_t i) {
				BuildKernel::reorderInteriors(
					i, ptNum - 1, mapidx.data(), interiors.getRawRepr(),
					rangeL.data(), rangeR.data(),
					splitDim.data(), splitVal.data(), parent.data());
			}
		);

		bufferPool->release<int>(std::move(interiors.rangeL));
		bufferPool->release<int>(std::move(interiors.rangeR));
		bufferPool->release<int>(std::move(interiors.splitDim));
		bufferPool->release<mfloat>(std::move(interiors.splitVal));
		bufferPool->release<int>(std::move(interiors.parent));

		interiors.rangeL = std::move(rangeL);
		interiors.rangeR = std::move(rangeR);
		interiors.splitDim = std::move(splitDim);
		interiors.splitVal = std::move(splitVal);
		interiors.parent = std::move(parent);


		parlay::parallel_for(0, ptNum,
			[&](size_t i) {
				BuildKernel::remapLeafParents(
					i, ptNum, mapidx.data(), leaves.getRawRepr());
			}
		);

		bufferPool->release<int>(std::move(mapidx));
#ifdef ENABLE_MERKLE
		// calc node hash
		parlay::parallel_for(0, ptNum,
			[&](size_t i) {
				BuildKernel::calcInteriorHash(i, ptNum, leaves.getRawRepr(),
				interiors.getRawRepr(), visitCount.data());
			}
		);
#endif
	}

	void PMKDTree::buildIncrement(const vector<vec3f>& ptsAdd) {
		size_t ptNum = primSize();
		size_t sizeInc = ptsAdd.size();
		// note: memory allocation can be async
		auto binIdx = bufferPool->acquire<int>(sizeInc);
		auto primIdx = bufferPool->acquire<int>(sizeInc);
		auto morton = bufferPool->acquire<MortonType>(sizeInc);
		auto mortonSorted = bufferPool->acquire<MortonType>(sizeInc);
		auto ptsAddSorted = bufferPool->acquire<vec3f>(sizeInc);

		// get scene boundary
		//sceneBoundary.merge(reduce<AABB>(ptsAdd, MergeOp()));
		//assert(globalBoundary.include(sceneBoundary));

		sortPts(ptsAdd, ptsAddSorted, primIdx, morton);
		// sort morton
		parlay::parallel_for(0, sizeInc,
			[&](size_t i) { mortonSorted[i] = morton[primIdx[i]]; }
		);
		bufferPool->release<MortonType>(std::move(morton));

		auto nodeMgrDevice = nodeMgr->getDeviceHandle();
		// find leaf bin
		int maxBin = -1;
		parlay::parallel_for(0, sizeInc,
			[&](size_t i) {
				UpdateKernel::findLeafBin(
					i, sizeInc, ptsAddSorted.data(), primSize(),
					nodeMgrDevice, binIdx.data());
			});
		maxBin = parlay::reduce(binIdx, parlay::maximum<int>());

		auto getMortonCode = [&](int gi) {
			if (gi >= ptNum) return mortonSorted[gi - ptNum].code;

			int iBatch, _offset;
			transformLeafIdx(gi, nodeMgrDevice.sizesAcc, nodeMgrDevice.numBatches, iBatch, _offset);
			return nodeMgrDevice.leavesBatch[iBatch].morton[_offset].code;
			};

		// get leafIdx
		auto leafIdxLeafSorted = parlay::remove_duplicate_integers(binIdx, maxBin + 1);
		// sort binIdx
		auto binIdxSorted = bufferPool->acquire<int>(sizeInc);
		parlay::parallel_for(0, sizeInc, [&](uint32_t i) {primIdx[i] = i;});
		parlay::integer_sort_inplace(primIdx, [&](const auto& idx) {return uint32_t(binIdx[idx]);});
		parlay::parallel_for(0, sizeInc, [&](size_t i) {binIdxSorted[i] = binIdx[primIdx[i]]; });
		parlay::parallel_for(0, sizeInc, [&](uint32_t i) {primIdx[i] += ptNum;});
		//bufferPool->release<int>(std::move(binIdx));


		size_t batchLeafSize = sizeInc + leafIdxLeafSorted.size();
		// allocate memory for final insertion
		// note: can be async
		Leaves leaves;
		leaves.resizePartial(batchLeafSize);
		leaves.treeLocalRangeR.resize(batchLeafSize);
		leaves.derivedFrom.resize(batchLeafSize);

		auto finalPrimIdx = bufferPool->acquire<int>(batchLeafSize);
		auto& finalBinIdx = leaves.derivedFrom;
		// parallel merge
		mergeZip(primIdx, leafIdxLeafSorted, binIdxSorted, finalPrimIdx, finalBinIdx,
			[&](const auto& idx1, const auto& idx2) {
				int bin1 = idx1 >= ptNum ? binIdx[idx1 - ptNum] : idx1;
				int bin2 = idx2 >= ptNum ? binIdx[idx2 - ptNum] : idx2;
				return bin1 < bin2 ||
					(bin1 == bin2 && getMortonCode(idx1) < getMortonCode(idx2));
			});
		// fmt::print("ptNum: {}\n", ptNum);
		// fmt::print("primIdx: {}\n", primIdx);
		// fmt::print("leafIdxLeafSorted: {}\n", leafIdxLeafSorted);
		// fmt::print("binIdx: {}\n", binIdx);
		// fmt::print("finalBinIdx: {}\n", finalBinIdx);

		bufferPool->release<int>(std::move(binIdx));
		bufferPool->release<int>(std::move(binIdxSorted));

		// allocate memory for final insertion
		// note: can be async

		Interiors interiors;
		interiors.resize(batchLeafSize - 1);

		auto mapidx = bufferPool->acquire<int>(batchLeafSize - 1);
		auto metrics = bufferPool->acquire<uint8_t>(batchLeafSize - 1);
		auto splitDim = bufferPool->acquire<int>(batchLeafSize - 1);
		auto splitVal = bufferPool->acquire<mfloat>(batchLeafSize - 1);
		auto parent = bufferPool->acquire<int>(batchLeafSize - 1);

		// build aid
		auto visitCount = std::vector<AtomicCount>(batchLeafSize - 1);
		auto innerBuf = bufferPool->acquire<int>(batchLeafSize - 1, 0);
		auto leafBuf = bufferPool->acquire<int>(batchLeafSize, 0);

		auto ptsAddFinal = bufferPool->acquire<vec3f>(batchLeafSize);
		auto treeLocalRangeL = bufferPool->acquire<int>(batchLeafSize);
		auto interiorCount = bufferPool->acquire<int>(leafIdxLeafSorted.size());
		auto interiorToLeafIdx = bufferPool->acquire<int>(batchLeafSize - leafIdxLeafSorted.size());

		// auto& tempIdx = primIdx;
		// tempIdx.resize(batchLeafSize);
		// auto& finalPrimIdx = binIdx;
		// finalPrimIdx.resize(batchLeafSize);

		// parlay::parallel_for(0, batchLeafSize,
		//     [&](size_t i) { tempIdx[i] = i; }
		// );
		// // sort by combinedBinIdx
		// parlay::stable_integer_sort_inplace(
		// 	tempIdx,
		// 	[&](const auto& idx) {return static_cast<uint32_t>(combinedBinIdx[idx]);});

		// parlay::parallel_for(0, batchLeafSize,
		//     [&](size_t i) { finalPrimIdx[i] = combinedPrimIdx[tempIdx[i]]; }
		// );
		// bufferPool->release<int>(std::move(combinedPrimIdx));

		// // set derivedFrom
		// auto& finalBinIdx = leaves.derivedFrom;
		// parlay::parallel_for(0, batchLeafSize,
		// 	[&](size_t i) { finalBinIdx[i] = combinedBinIdx[tempIdx[i]]; }
		// );
		// bufferPool->release<int>(std::move(tempIdx));
		// bufferPool->release<int>(std::move(combinedBinIdx));

		// plan 1
		//mTimer("设置pt和morton", [&] {
			// set final points to add
		parlay::parallel_for(0, batchLeafSize,
			[&](size_t i) {
				int gi = finalPrimIdx[i];
				if (gi >= ptNum) ptsAddFinal[i] = ptsAddSorted[gi - ptNum];
				else {
					int iBatch, _offset;
					transformLeafIdx(gi, nodeMgrDevice.sizesAcc, nodeMgrDevice.numBatches, iBatch, _offset);
					ptsAddFinal[i] = nodeMgrDevice.ptsBatch[iBatch][_offset];
				}
			}
		);
		// set final mortons to add, set removal
		parlay::parallel_for(0, batchLeafSize,
			[&](size_t i) {
				int gi = finalPrimIdx[i];
				if (gi >= ptNum) leaves.morton[i] = mortonSorted[gi - ptNum];
				else {
					int iBatch, _offset;
					transformLeafIdx(gi, nodeMgrDevice.sizesAcc, nodeMgrDevice.numBatches, iBatch, _offset);
					leaves.morton[i] = nodeMgrDevice.leavesBatch[iBatch].morton[_offset];
					// set removal
					int r = nodeMgrDevice.leavesBatch[iBatch].replacedBy[_offset];
					leaves.replacedBy[i] = r;
				}
			}
		);
		//});

		// plan 2
		// mTimer("设置pt和morton", [&] {
		// auto finalPrimIdxNew = bufferPool->acquire<int>(sizeInc);
		// auto finalPrimIdxOriginal = bufferPool->acquire<int>(leafIdxLeafSorted.size());

		// int i1 = 0, i2 = 0;
		// for (size_t i = 0; i < batchLeafSize; i++) {
		// 	int gi = finalPrimIdx[i];
		// 	if (gi >= ptNum) finalPrimIdxNew[i1++] = i;
		// 	else finalPrimIdxOriginal[i2++] = i;
		// }
		// parlay::parallel_for(0, sizeInc,
		// 	[&](size_t i) {
		// 			int idx = finalPrimIdxNew[i];
		// 			int gi = finalPrimIdx[idx];
		// 			ptsAddFinal[idx] = ptsAddSorted[gi - ptNum];
		// 			leaves.morton[idx] = mortonSorted[gi - ptNum];
		// 		}
		// );

		// parlay::parallel_for(0, leafIdxLeafSorted.size(),
		// 	[&](size_t i) {
		// 		int idx = finalPrimIdxOriginal[i];
		// 		int gi = finalPrimIdx[idx];

		// 		int iBatch, _offset;
		// 		transformLeafIdx(gi, nodeMgrDevice.sizesAcc, nodeMgrDevice.numBatches, iBatch, _offset);
		// 		leaves.morton[idx] = nodeMgrDevice.leavesBatch[iBatch].morton[_offset];
		// 		ptsAddFinal[idx] = nodeMgrDevice.ptsBatch[iBatch][_offset];
		// 		// set removal
		// 		int r = nodeMgrDevice.leavesBatch[iBatch].replacedBy[_offset];
		// 		leaves.replacedBy[idx] = r;
		// 	}
		// );
		// bufferPool->release<int>(std::move(finalPrimIdxNew));
		// bufferPool->release<int>(std::move(finalPrimIdxOriginal));
		// });

		bufferPool->release<vec3f>(std::move(ptsAddSorted));
		// note: leaf hash calculation can be boost
		// by recording leafBinIdx2finalPrimIdx and copying hash
		// to avoid some costly hash calculation
		bufferPool->release<int>(std::move(finalPrimIdx));
		bufferPool->release<MortonType>(std::move(mortonSorted));

		// set bin count
		parlay::parallel_for(0, leafIdxLeafSorted.size(), [&](size_t i) {
			const auto& [_l, _r] = std::equal_range(finalBinIdx.begin(), finalBinIdx.end(), leafIdxLeafSorted[i]);
			interiorCount[i] = _r - _l - 1;
			});
		parlay::scan_inplace(interiorCount);


		// set tree local range
		parlay::parallel_for(0, batchLeafSize, [&](size_t i) {
			int j = std::lower_bound(leafIdxLeafSorted.begin(), leafIdxLeafSorted.end(), finalBinIdx[i]) - leafIdxLeafSorted.begin();
			treeLocalRangeL[i] = interiorCount[j] + j;
			leaves.treeLocalRangeR[i] = j < leafIdxLeafSorted.size() - 1 ? interiorCount[j + 1] + j + 1 : batchLeafSize;
			});

		// calc metrics
		parlay::parallel_for(0, interiorToLeafIdx.size(), [&](size_t i) {
			int j = std::upper_bound(interiorCount.begin(), interiorCount.end(), i) - interiorCount.begin() - 1;
			interiorToLeafIdx[i] = i + j;
			});

		parlay::parallel_for(0, interiorToLeafIdx.size(), [&](size_t i) {
			DynamicBuildKernel::calcBuildMetrics(i, interiorToLeafIdx.size(), globalBoundary,
			leaves.morton.data(), interiorToLeafIdx.data(),
			metrics.data(), interiors.splitDim.data(), interiors.splitVal.data());
			});

		//---------------------------------------------------------------------
		// build interiors

		BuildAid aid{ metrics.data(), visitCount.data(), innerBuf.data(),leafBuf.data() };

		// build interior nodes
		// note: the following optimization does not prove better on CPU
		// may be better on GPU
		// if (config.optimize) {
		// 	int* range[2] = { interiors.rangeL.data(),interiors.rangeR.data() };
		// 	parlay::parallel_for(0, batchLeafSize,
		// 		[&](size_t i) {
		// 			BuildKernel::buildInteriors_opt(
		// 				i, batchLeafSize, leaves.getRawRepr(),
		// 				range, interiors.splitDim.data(), interiors.splitVal.data(),
		// 				interiors.parentSplitDim.data(), interiors.parentSplitVal.data(),
		// 				aid);
		// 		}
		// 	);
		// }
		// else {
		parlay::parallel_for(0, batchLeafSize,
			[&](size_t i) {
				DynamicBuildKernel::buildInteriors(
					i, batchLeafSize, treeLocalRangeL.data(), leaves.getRawRepr(), interiors.getRawRepr(), aid);
			}
		);
		//}
		bufferPool->release<int>(std::move(treeLocalRangeL));
		bufferPool->release<uint8_t>(std::move(metrics));

		// calculate new indices for interiors
		auto& segLen = leafBuf;
		leaves.segOffset = parlay::scan(segLen).first;

		parlay::parallel_for(0, interiorCount.size() - 1, [&](size_t i) {
			DynamicBuildKernel::interiorMapIdxInit(i, interiorCount.size(), batchLeafSize, interiorCount.data(), mapidx.data());
			});


		parlay::parallel_for(0, interiorToLeafIdx.size(),
			[&](size_t i) {
				DynamicBuildKernel::calcInteriorNewIdx(
					i, interiorToLeafIdx.size(), interiorToLeafIdx.data(),
					leaves.getRawRepr(), interiors.getRawRepr(), aid.segLen, aid.leftLeafCount, mapidx.data());
			}
		);
		bufferPool->release<int>(std::move(interiorToLeafIdx));

		// reorder interiors
		auto& rangeL = innerBuf;
		auto& rangeR = leafBuf;
		rangeR.resize(batchLeafSize - 1);

		parlay::parallel_for(0, batchLeafSize - 1,
			[&](size_t i) {
				DynamicBuildKernel::reorderInteriors(
					i, batchLeafSize - 1, mapidx.data(), interiors.getRawRepr(),
					rangeL.data(), rangeR.data(),
					splitDim.data(), splitVal.data(), parent.data());
			}
		);
		bufferPool->release<int>(std::move(interiors.rangeL));
		bufferPool->release<int>(std::move(interiors.rangeR));
		bufferPool->release<int>(std::move(interiors.splitDim));
		bufferPool->release<mfloat>(std::move(interiors.splitVal));
		bufferPool->release<int>(std::move(interiors.parent));

		interiors.rangeL = std::move(rangeL);
		interiors.rangeR = std::move(rangeR);
		interiors.splitDim = std::move(splitDim);
		interiors.splitVal = std::move(splitVal);
		interiors.parent = std::move(parent);

		parlay::parallel_for(0, batchLeafSize,
			[&](size_t i) {
				DynamicBuildKernel::remapLeafParents(
					i, batchLeafSize, mapidx.data(), leaves.getRawRepr());
			}
		);

		bufferPool->release<int>(std::move(mapidx));

		// set replacedBy of leafbins and parent of subtrees' roots
		parlay::parallel_for(0, leafIdxLeafSorted.size(), [&](size_t i) {
			int gi = leafIdxLeafSorted[i];
			int iBatch, _offset;
			transformLeafIdx(gi, nodeMgrDevice.sizesAcc, nodeMgrDevice.numBatches, iBatch, _offset);
			auto& binLeaves = nodeMgrDevice.leavesBatch[iBatch];
			const auto& binInteriors = nodeMgrDevice.interiorsBatch[iBatch];

			binLeaves.replacedBy[_offset] = ptNum + interiorCount[i] + i;

			int parentCode = ((gi - _offset) << 1) + binLeaves.parent[_offset];
			interiors.parent[interiorCount[i]] = -parentCode;
			});

#ifdef ENABLE_MERKLE
		// calculate node hash
		parlay::parallel_for(0, batchLeafSize,
			[&](size_t i) {
				BuildKernel::calcLeafHash(i, batchLeafSize, ptsAddFinal.data(), leaves.replacedBy.data(), leaves.hash.data());
			}
		);
		parlay::parallel_for(0, batchLeafSize,
			[&](size_t i) {
				DynamicBuildKernel::calcInteriorHash_Batch(i, batchLeafSize,
				leaves.getRawRepr(), interiors.getRawRepr());
			}
		);
#endif
		// revert removal of bins to insert
		parlay::parallel_for(0, leafIdxLeafSorted.size(),
			[&](size_t i) {
				UpdateKernel::revertRemoval(i, leafIdxLeafSorted.size(), leafIdxLeafSorted.data(), nodeMgrDevice);
			}
		);
#ifdef ENABLE_MERKLE
		// calculate node hash
		parlay::parallel_for(0, interiorCount.size(),
			[&](size_t i) {
				DynamicBuildKernel::calcInteriorHash_Upper(i, interiorCount.size(), interiorCount.data(), leafIdxLeafSorted.data(),
				interiors.getRawRepr(), nodeMgrDevice);
			}
		);
		// auto rootHash = getRootHash();
		// printf("root hash: %d,%d,%d\n", int(rootHash.byte[0]), int(rootHash.byte[1]), int(rootHash.byte[2]));
		// printf("\n");
#endif
		bufferPool->release<int>(std::move(interiorCount));

		nodeMgr->append(std::move(leaves), std::move(interiors), std::move(ptsAddFinal));
	}

	void PMKDTree::buildIncrement_v2(const vector<vec3f>& ptsAdd) {
		auto& leaves = nodeMgr->getLeaves(0);
		auto& interiors = nodeMgr->getInteriors(0);
		auto& pts = nodeMgr->getPtsBatch(0);

		size_t ptNum = leaves.size();
		size_t sizeInc = ptsAdd.size();

		// note: memory allocation can be async
		auto mortonAdd = bufferPool->acquire<MortonType>(sizeInc);
		auto mortonAddSorted = bufferPool->acquire<MortonType>(sizeInc);
		auto primIdxAdd = bufferPool->acquire<int>(sizeInc);
		auto ptsAddSorted = bufferPool->acquire<vec3f>(sizeInc);

		vector<vec3f> ptsSortedFinal(ptNum + sizeInc);
		vector<MortonType> mortonSortedFinal(ptNum + sizeInc);
#ifdef ENABLE_MERKLE
		vector<hash_t> hashAdd(sizeInc);
		vector<hash_t> hashFinal(ptNum + sizeInc);
#endif

		auto primIdx = bufferPool->acquire<int>(ptNum);

		// get scene boundary
		//sceneBoundary.merge(reduce<AABB>(ptsAdd, MergeOp()));
		//assert(globalBoundary.include(sceneBoundary));

		sortPts(ptsAdd, ptsAddSorted, primIdxAdd, mortonAdd);
		parlay::parallel_for(0, sizeInc, [&](size_t i) { mortonAddSorted[i] = mortonAdd[primIdxAdd[i]];});

#ifdef ENABLE_MERKLE
		parlay::parallel_for(0, sizeInc, [&](size_t i) { BuildKernel::calcLeafHash(i, sizeInc, ptsAddSorted.data(), hashAdd.data());});
#endif
		bufferPool->release<MortonType>(std::move(mortonAdd));

		parlay::parallel_for(0, sizeInc,
			[&](size_t i) { primIdxAdd[i] = ptNum + i; }
		);
		parlay::parallel_for(0, ptNum,
			[&](size_t i) { primIdx[i] = i; }
		);

		auto primIdxFinal = parlay::merge(primIdx, primIdxAdd,
			[&](int _i, int _j) {
				MortonType mi = _i < ptNum ? leaves.morton[_i] : mortonAddSorted[_i - ptNum];
				MortonType mj = _j < ptNum ? leaves.morton[_j] : mortonAddSorted[_j - ptNum];
				return mi.code < mj.code;
			}
		);
		bufferPool->release<int>(std::move(primIdx));
		bufferPool->release<int>(std::move(primIdxAdd));


		parlay::parallel_for(0, ptNum + sizeInc, [&](size_t i) {
			int j = primIdxFinal[i];
			ptsSortedFinal[i] = j < ptNum ? pts[j] : ptsAddSorted[j - ptNum];
			});
		bufferPool->release<vec3f>(std::move(ptsAddSorted));

		parlay::parallel_for(0, ptNum + sizeInc, [&](size_t i) {
			int j = primIdxFinal[i];
			mortonSortedFinal[i] = j < ptNum ? leaves.morton[j] : mortonAddSorted[j - ptNum];
	});
		bufferPool->release<MortonType>(std::move(mortonAddSorted));

#ifdef ENABLE_MERKLE
		parlay::parallel_for(0, ptNum + sizeInc, [&](size_t i) {
			int j = primIdxFinal[i];
			hashFinal[i] = j < ptNum ? leaves.hash[j] : hashAdd[j - ptNum];
			});
		hashAdd.clear();
#endif
		primIdxFinal.clear();

		leaves.morton = std::move(mortonSortedFinal);
		pts = std::move(ptsSortedFinal);
		ptNum = leaves.size();

		leaves.replacedBy.resize(ptNum, 0);
		leaves.parent.resize(ptNum);
#ifdef ENABLE_MERKLE
		leaves.hash = std::move(hashFinal);
#endif

		interiors.resize(ptNum - 1);

		buildStatic_LeavesReady(leaves, interiors);
		nodeMgr->refitBatch(0);
}

	PMKD_PrintInfo PMKDTree::print(bool verbose) const {
		if (isStatic) return printStatic(verbose);
		return printDynamic(verbose);
	}

	PMKD_PrintInfo PMKDTree::printStatic(bool verbose) const {
		assert(isStatic);

		size_t ptNum = primSize();

		PMKD_PrintInfo info;
		info.leafNum = primSize();
		if (ptNum == 0) return info;

		auto hostNodeMgr = nodeMgr->copyToHost();
		const auto& leaves = hostNodeMgr.leavesBatch[0];
		const auto& interiors = hostNodeMgr.interiorsBatch[0];

		using PNode = std::pair<int, int>;  // <leafBin, globalIdx>, where leaf's globalIdx += ptNum

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
			info.leafPoints = getStoredPoints();
			info.splitDim.assign(interiors.splitDim.begin(), interiors.splitDim.end());
			info.splitVal.assign(interiors.splitVal.begin(), interiors.splitVal.end());
		}
		return info;
	}

	PMKD_PrintInfo PMKDTree::printDynamic(bool verbose) const {
		assert(!isStatic);

		size_t ptNum = primSize();

		PMKD_PrintInfo info;
		info.leafNum = primSize();
		if (ptNum == 0) return info;

		auto hostNodeMgr = nodeMgr->copyToHost();

		using PNode = std::pair<int, int>;  // <leafBin, globalIdx>, where leaf's globalIdx += ptNum

		auto left = [&](const PNode& pnode) ->PNode {
			auto& [leafBinIdx, gIdx] = pnode;

			if (gIdx >= ptNum) return { -1,-1 };  // leaf has no child

			int nextLeafBinIdx = leafBinIdx;
			while (nextLeafBinIdx < ptNum) {
				int iBatch, localLeafIdx;
				transformLeafIdx(nextLeafBinIdx, hostNodeMgr.sizesAcc.data(),
					hostNodeMgr.leavesBatch.size(), iBatch, localLeafIdx);

				const auto& leaves = hostNodeMgr.leavesBatch[iBatch];
				const auto& interiors = hostNodeMgr.interiorsBatch[iBatch];
				int rBound = iBatch > 0 ? leaves.treeLocalRangeR[localLeafIdx] : leaves.size();
				assert(localLeafIdx < rBound - 1);

				int interiorIdx = iBatch > 0 ? gIdx - hostNodeMgr.sizesAcc[iBatch - 1] : gIdx;
				if (nextLeafBinIdx != leafBinIdx) {
					assert(iBatch > 0);
					interiorIdx = leaves.segOffset[localLeafIdx];
					int nextGlobal = interiorIdx + hostNodeMgr.sizesAcc[iBatch - 1];
					return { nextLeafBinIdx,nextGlobal };
				}

				int R = leaves.segOffset[localLeafIdx + 1];
				int nextLocal = interiorIdx + 1;  // if interiorIdx < R - 1

				if (interiorIdx < R - 1) {
					// left child is an nterior node
					int nextGlobal = iBatch > 0 ? nextLocal + hostNodeMgr.sizesAcc[iBatch - 1] : nextLocal;
					return { nextLeafBinIdx,nextGlobal };
				}

				assert(interiorIdx == R - 1);
				// left child is potentially a leaf
				int globalSubstitute = leaves.replacedBy[localLeafIdx];
				if (globalSubstitute == 0) // this leaf is valid
					return { nextLeafBinIdx, nextLeafBinIdx + ptNum };

				nextLeafBinIdx = globalSubstitute;
			}
			fmt::print("error!\n");
			assert(0);
			return { leafBinIdx,gIdx };
			};

		auto right = [&](const PNode& pnode) ->PNode {
			auto& [leafBinIdx, gIdx] = pnode;

			if (gIdx >= ptNum) return { -1,-1 };  // leaf has no child

			int nextLeafBinIdx = leafBinIdx;
			while (nextLeafBinIdx < ptNum) {
				int iBatch, localLeafIdx;
				transformLeafIdx(nextLeafBinIdx, hostNodeMgr.sizesAcc.data(),
					hostNodeMgr.leavesBatch.size(), iBatch, localLeafIdx);

				const auto& leaves = hostNodeMgr.leavesBatch[iBatch];
				const auto& interiors = hostNodeMgr.interiorsBatch[iBatch];

				int rBound = iBatch > 0 ? leaves.treeLocalRangeR[localLeafIdx] : leaves.size();

				int interiorIdx = iBatch > 0 ? gIdx - hostNodeMgr.sizesAcc[iBatch - 1] : gIdx;

				if (nextLeafBinIdx != leafBinIdx) {
					assert(iBatch > 0);
					interiorIdx = leaves.segOffset[localLeafIdx];
					int nextGlobal = interiorIdx + hostNodeMgr.sizesAcc[iBatch - 1];
					return { nextLeafBinIdx,nextGlobal };
				}

				int R = leaves.segOffset[localLeafIdx + 1];
				assert(interiorIdx < R);

				int nextLocalBin = interiorIdx == R - 1 ?
					localLeafIdx + 1 : interiors.rangeR[interiorIdx + 1] + 1;
				int nextLocalIdx = leaves.segOffset[nextLocalBin];

				nextLeafBinIdx += nextLocalBin - localLeafIdx;
				if (nextLocalBin == rBound - 1) {
					//assert(nextLocalBin == nextLocalIdx);
					// right child is potentially a leaf
					int globalSubstitute = leaves.replacedBy[nextLocalBin];

					if (globalSubstitute == 0) // this leaf is valid
						return { nextLeafBinIdx, nextLeafBinIdx + ptNum };

					// leaf is replaced
					nextLeafBinIdx = globalSubstitute;
					continue;
				}
				else if (nextLocalIdx == leaves.segOffset[nextLocalBin + 1]) {
					// right child is potentially a leaf
					int globalSubstitute = leaves.replacedBy[nextLocalBin];

					if (globalSubstitute == 0) // this leaf is valid
						return { nextLeafBinIdx, nextLeafBinIdx + ptNum };

					// leaf is replaced
					nextLeafBinIdx = globalSubstitute;
					continue;
				}
				// right child is not a leaf
				int nextGlobal = iBatch > 0 ? nextLocalIdx + hostNodeMgr.sizesAcc[iBatch - 1] : nextLocalIdx;
				return { nextLeafBinIdx,nextGlobal };
			}
			fmt::print("error!\n");
			assert(0);
			return { leafBinIdx,gIdx };
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
		stk.clear();
		stk.push_back({ 0,0 });
		while (!stk.empty()) {
			p = stk[stk.size() - 1];
			stk.pop_back();
			info.preorderTraversal.push_back(p.second);

			auto r = right(p);
			if (!isInvalid(r)) stk.push_back(r);

			auto l = left(p);
			if (!isInvalid(l)) stk.push_back(l);
		}
		assert(info.preorderTraversal.size() == info.inorderTraversal.size());

		for (size_t i = 0;i < hostNodeMgr.leavesBatch.size();++i) {
			const auto& leaves = hostNodeMgr.leavesBatch[i];
			// get leaf mortons
			info.leafMortons.insert(info.leafMortons.end(), leaves.morton.begin(), leaves.morton.end());
			if (verbose) {
				info.leafPoints.insert(info.leafPoints.end(), hostNodeMgr.ptsBatch[i].begin(), hostNodeMgr.ptsBatch[i].end());

				const auto& interiors = hostNodeMgr.interiorsBatch[i];
				info.splitDim.insert(info.splitDim.end(), interiors.splitDim.begin(), interiors.splitDim.end());
				info.splitDim.push_back(-1);

				info.splitVal.insert(info.splitVal.end(), interiors.splitVal.begin(), interiors.splitVal.end());
				info.splitVal.push_back(0);
			}
		}

		// get morton metrics
		// info.metrics.resize(ptNum - 1);
		// parlay::parallel_for(0, ptNum - 1,
		// 	[&](size_t i) {
		// 		info.metrics[i] = MortonType::calcMetric(info.leafMortons[i], info.leafMortons[i + 1]);
		// 	}
		// );
		return info;
	}


	std::vector<vec3f> PMKDTree::getStoredPoints() const {
		auto dpts = nodeMgr->flattenPoints();

		std::vector<vec3f> pts(dpts.begin(), dpts.end());
		return pts;
	}

	QueryResponses PMKDTree::query(const vector<Query>& queries) const {
		if (queries.empty()) return QueryResponses(0);

		size_t nq = queries.size();
		QueryResponses responses(nq);

		vector<Query> queriesSorted;
		const Query* target = queries.data();
		// sort queries to improve cache friendlyness
		if (config.optimize) {
			queriesSorted = bufferPool->acquire<vec3f>(nq);
			sortPts(queries, queriesSorted, responses.queryIdx);
			target = queriesSorted.data();
		}

		if (isStatic) {
			assert(nodeMgr->numBatches() == 1);
			const auto& leaves = nodeMgr->getLeaves(0);
			const auto& interiors = nodeMgr->getInteriors(0);

			parlay::parallel_for(0, nq,
				[&](size_t i) {
					SearchKernel::searchPoints(
						i, nq, target, nodeMgr->getPtsBatch(0).data(), primSize(),
						interiors.getRawRepr(), leaves.getRawRepr(), AABB::worldBox(), responses.exist.data());
				}
			);
		}
		else {
			parlay::parallel_for(0, nq, [&](size_t i) {
				SearchKernel::searchPoints(i, nq, target, nodeMgr->getDeviceHandle(), primSize(), AABB::worldBox(), responses.exist.data());
				});
		}

		if (!queriesSorted.empty()) bufferPool->release(std::move(queriesSorted));
		return responses;
	}


	void PMKDTree::_query(const vector<RangeQuery>& queries, RangeQueryResponses& responses) const {
		size_t nq = queries.size();

		vector<RangeQuery> queriesSorted;
		const RangeQuery* target = queries.data();
		// note: sort queries as an optimization
		if (false) {
		//if (config.optimize) {
			auto centers = bufferPool->acquire<vec3f>(nq);
			auto morton = bufferPool->acquire<MortonType>(nq);

			parlay::parallel_for(0, nq,
				[&](size_t i) { centers[i] = queries[i].center(); }
			);
			parlay::parallel_for(0, nq,
				[&](size_t i) { BuildKernel::calcMortonCodes(i, nq, centers.data(), &globalBoundary, morton.data()); }
			);
			// note: there are multiple sorting algorithms to choose from
			parlay::integer_sort_inplace(
				responses.queryIdx, [&](const auto& idx) {return morton[idx].code;});

			queriesSorted.resize(nq);
			parlay::parallel_for(0, nq, [&](size_t i) {queriesSorted[i] = queries[responses.queryIdx[i]];});

			target = queriesSorted.data();

			bufferPool->release(std::move(centers));
			bufferPool->release(std::move(morton));
		}

		if (isStatic) {
			assert(nodeMgr->numBatches() == 1);
			const auto& leaves = nodeMgr->getLeaves(0);
			const auto& interiors = nodeMgr->getInteriors(0);

			parlay::parallel_for(0, nq,
				[&](size_t i) {
					SearchKernel::searchRanges(
						i, nq, target, nodeMgr->getPtsBatch(0).data(), primSize(),
						interiors.getRawRepr(), leaves.getRawRepr(),
						AABB::worldBox(), responses.getRawRepr());
				}
			);
		}
		else {
			parlay::parallel_for(0, nq, [&](size_t i) {
				SearchKernel::searchRanges(i, nq, target,
				nodeMgr->getDeviceHandle(), primSize(), AABB::worldBox(),
				responses.getRawRepr());
				});
		}
	}

	RangeQueryResponses PMKDTree::query(const vector<RangeQuery>& queries) const {
		if (queries.empty()) return RangeQueryResponses(0);

		RangeQueryResponses responses(queries.size());

		mTimer("Range Search Time", [&] {this->_query(queries, responses);});

		return responses;
	}

#ifdef ENABLE_MERKLE
	hash_t PMKDTree::getRootHash() const {
		return nodeMgr->getInteriors(0).hash[0];
	}

	VerifiableRangeQueryResponses
		PMKDTree::verifiableQuery(const vector<RangeQuery>& queries) const {
		if (queries.empty()) return VerifiableRangeQueryResponses();

		size_t nq = queries.size();
		VerifiableRangeQueryResponses responses(nq);

		vector<RangeQuery> queriesSorted;
		const RangeQuery* target = queries.data();
		// sort queries as an optimization
		if (config.optimize) {
			auto centers = bufferPool->acquire<vec3f>(nq);
			auto morton = bufferPool->acquire<MortonType>(nq);

			parlay::parallel_for(0, nq,
				[&](size_t i) { centers[i] = queries[i].center(); }
			);

			parlay::parallel_for(0, nq,
				[&](size_t i) { BuildKernel::calcMortonCodes(i, nq, centers.data(), &globalBoundary, morton.data()); }
			);
			// note: there are multiple sorting algorithms to choose from
			parlay::integer_sort_inplace(
				responses.queryIdx, [&](const auto& idx) {return morton[idx].code;});

			queriesSorted.resize(nq);
			parlay::parallel_for(0, nq, [&](size_t i) {queriesSorted[i] = queries[responses.queryIdx[i]];});

			target = queriesSorted.data();

			bufferPool->release(std::move(centers));
			bufferPool->release(std::move(morton));
		}

		NodeMgrDevice nodeMgrDevice = nodeMgr->getDeviceHandle();
		size_t ptNum = primSize();
		// pass 1
		parlay::parallel_for(0, nq, [&](size_t i) {
			SearchKernel::searchRangesVerifiable_step1(
				i, nq, target, nodeMgrDevice, ptNum,
				responses.fOffset.data(), responses.mOffset.data(), responses.hOffset.data()
			);
			});
		int fSize = parlay::scan_inplace(responses.fOffset);
		int mSize = parlay::scan_inplace(responses.mOffset);
		int hSize = parlay::scan_inplace(responses.hOffset);
		responses.initVerificationSet(fSize, mSize, hSize);

		// pass 2
		parlay::parallel_for(0, nq, [&](size_t i) {
			SearchKernel::searchRangesVerifiable_step2(
				i, nq, target, nodeMgrDevice, ptNum,
				responses.fOffset.data(), responses.mOffset.data(), responses.hOffset.data(),
				responses.vs.fNodes.getRawRepr(), responses.vs.mNodes.getRawRepr(), responses.vs.hNodes.getRawRepr()
			);
			});
		return responses;
	}
#endif

	void PMKDTree::rebuildUponInsert(const vector<vec3f>& ptsAdd) {

	}

	void PMKDTree::rebuildUponRemove(const vector<vec3f>& ptsRemove) {

	}

	// rebuild strategy
	bool PMKDTree::needRebuild(int nToDInsert, int nToRemove) const {
		int nBatches = nodeMgr->numBatches();
		if (nToDInsert > 0 && nBatches + 1 > config.maxNumBatches) return true;

		float nValid = primSize() - nTotalRemoved;
		float ratioI = (nToDInsert + nTotalDInserted) / nValid;
		float ratioR = (nToRemove + nTotalRemoved) / nValid;
		return ratioI >= config.maxDInsertedRatio || ratioR >= config.maxRemovedRatio;
	}

	void PMKDTree::insert(const vector<vec3f>& ptsAdd) {
		if (ptsAdd.empty()) return;

		int nStored = primSize();
		if (nStored == 0) {
			firstInsert(ptsAdd);
			return;
		}

		if (needRebuild(ptsAdd.size(), 0)) {
			rebuildUponInsert(ptsAdd);
			isStatic = true;
		}
		else {
			isStatic = false;
			buildIncrement(ptsAdd);
			nTotalDInserted += ptsAdd.size();
		}
	}

	void PMKDTree::insert_v2(const vector<vec3f>& ptsAdd) {
		if (ptsAdd.empty()) return;

		assert(isStatic);
		buildIncrement_v2(ptsAdd);
	}

	void PMKDTree::firstInsert(const vector<vec3f>& pts) {
		if (pts.empty()) return;

		destroy();
		isStatic = true;
		buildStatic(pts);
	}

	void PMKDTree::remove(const vector<vec3f>& ptsRemove) {
		if (ptsRemove.empty()) return;

		size_t nq = ptsRemove.size();

		vector<vec3f> ptsRemoveSorted;
		const vec3f* target = ptsRemove.data();

		auto binIdx = bufferPool->acquire<int>(nq);

		// sort queries to improve cache friendlyness
		if (config.optimize) {
			ptsRemoveSorted = bufferPool->acquire<vec3f>(nq);
			sortPts(ptsRemove, ptsRemoveSorted);

			target = ptsRemoveSorted.data();
		}

		auto nodeMgrDeviceHandle = nodeMgr->getDeviceHandle();

		if (isStatic) {
			assert(nodeMgr->numBatches() == 1);
			const auto& leaves = nodeMgr->getLeaves(0);
			const auto& interiors = nodeMgr->getInteriors(0);

			parlay::parallel_for(0, nq,
				[&](size_t i) {
					UpdateKernel::removePoints_step1(i, nq, target, nodeMgr->getPtsBatch(0).data(), primSize(),
					interiors.getRawRepr(), leaves.getRawRepr(), binIdx.data());
		}
			);

#ifdef ENABLE_MERKLE
			parlay::parallel_for(0, nq, [&](size_t i) {
				UpdateKernel::calcSelectedLeafHash(i, nq, binIdx.data(), nodeMgrDeviceHandle);
				});
#endif

			parlay::parallel_for(0, nq, [&](size_t i) {
				UpdateKernel::removePoints_step2(i, nq, primSize(), binIdx.data(), leaves.getRawRepr(), interiors.getRawRepr());
				});
		}
		else {
			parlay::parallel_for(0, nq, [&](size_t i) {
				UpdateKernel::removePoints_step1(i, nq, target, nodeMgrDeviceHandle, primSize(), binIdx.data());
				}
			);

#ifdef ENABLE_MERKLE
			parlay::parallel_for(0, nq, [&](size_t i) {
				UpdateKernel::calcSelectedLeafHash(i, nq, binIdx.data(), nodeMgrDeviceHandle);
				});
#endif

			parlay::parallel_for(0, nq, [&](size_t i) {
				UpdateKernel::removePoints_step2(i, nq, binIdx.data(), nodeMgrDeviceHandle);
				});
				}
		if (!ptsRemoveSorted.empty()) bufferPool->release(std::move(ptsRemoveSorted));
		bufferPool->release(std::move(binIdx));

		isStatic = false;
	}

	void PMKDTree::remove_v2(const vector<vec3f>& ptsRemove) {
		assert(isStatic);

		if (ptsRemove.empty()) return;

		size_t nq = ptsRemove.size();

		vector<vec3f> ptsRemoveSorted;
		const vec3f* target = ptsRemove.data();

		size_t ptNum = primSize();
		size_t ptNumNew = ptNum - nq;

		// note: memory allocation can be async
		auto primIdxNew = bufferPool->acquire<int>(ptNumNew);
		vector<vec3f> ptsFinal(ptNumNew);
		vector<MortonType> mortonFinal(ptNumNew);
#ifdef ENABLE_MERKLE
		vector<hash_t> hashFinal(ptNumNew);
#endif

		// sort queries to improve cache friendlyness
		if (config.optimize) {
			ptsRemoveSorted = bufferPool->acquire<vec3f>(nq);
			sortPts(ptsRemove, ptsRemoveSorted);

			target = ptsRemoveSorted.data();
		}

		assert(nodeMgr->numBatches() == 1);
		auto& leaves = nodeMgr->getLeaves(0);
		auto& interiors = nodeMgr->getInteriors(0);
		auto& pts = nodeMgr->getPtsBatch(0);

		parlay::parallel_for(0, nq,
			[&](size_t i) {
				UpdateKernel::removePoints_v2_step1(i, nq, target, pts.data(), ptNum,
				interiors.getRawRepr(), leaves.getRawRepr());
			}
		);

		if (!ptsRemoveSorted.empty()) bufferPool->release(std::move(ptsRemoveSorted));

		// 构造新的leaves，去除binIdx部分
		// 忽略这部分耗时(todo: better algorithm)
		mTimer("忽略构造primIdxNew耗时", [&] {
			auto idxs = parlay::iota<int>(ptNum);
			parlay::filter_into_uninitialized(idxs, primIdxNew, [&](int e) {return leaves.replacedBy[e] != -1;});
			// size_t j = 0;
			// for (size_t i = 0; i < ptNum; i++) {
			// 	if (leaves.replacedBy[i] == 0) {
			// 		primIdxNew[j++] = i;
			// 	}
			// }
			});

		parlay::parallel_for(0, ptNumNew, [&](size_t i) {
			ptsFinal[i] = pts[primIdxNew[i]];
	});

		parlay::parallel_for(0, ptNumNew, [&](size_t i) {
			mortonFinal[i] = leaves.morton[primIdxNew[i]];
			});
#ifdef ENABLE_MERKLE
		parlay::parallel_for(0, ptNumNew, [&](size_t i) {
			hashFinal[i] = leaves.hash[primIdxNew[i]];
			});
#endif
		bufferPool->release(std::move(primIdxNew));

		// 		leaves.morton = std::move(mortonFinal);
		// #ifdef ENABLE_MERKLE
		// 		leaves.hash = std::move(hashFinal);
		// #endif
		// 		leaves.replacedBy.clear();
		// 		leaves.replacedBy.resize(ptNumNew, 0);
		// 		leaves.parent.resize(ptNumNew);
		// 		interiors.resize(ptNumNew - 1);

		// 		pts = std::move(ptsFinal);

		// 		// get scene boundary
		// 		sceneBoundary = reduce<AABB>(pts, MergeOp());
		// 		globalBoundary = sceneBoundary;

		// 		buildStatic_LeavesReady(leaves, interiors);
		// 		nodeMgr->refitBatch(0);


		Leaves leavesNew = std::move(leaves);
		leavesNew.replacedBy.clear();
		leavesNew.replacedBy.resize(ptNumNew, 0);
		leavesNew.parent.resize(ptNumNew);

		Interiors interiorsNew = std::move(interiors);
		interiorsNew.resize(ptNumNew - 1);

		destroy();
		isStatic = true;

		// get scene boundary
		//sceneBoundary = reduce<AABB>(ptsFinal, MergeOp());
		//globalBoundary.merge(sceneBoundary);

		leavesNew.morton = std::move(mortonFinal);
#ifdef ENABLE_MERKLE
		leavesNew.hash = std::move(hashFinal);
#endif

		buildStatic_LeavesReady(leavesNew, interiorsNew);
		nodeMgr->append(std::move(leavesNew), std::move(interiorsNew), std::move(ptsFinal));
	}
}