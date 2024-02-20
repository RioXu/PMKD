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
		isStatic = false;
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
	}

	void PMKDTree::buildStatic(const vector<vec3f>& pts) {
		size_t ptNum = pts.size();

		// get scene boundary
		sceneBoundary = reduce<AABB>(pts, MergeOp());
		globalBoundary.merge(sceneBoundary);

		auto primIdx = bufferPool->acquire<int>(ptNum);
		parlay::parallel_for(0, ptNum, [&](size_t i) {primIdx[i] = i;});
		auto _morton = bufferPool->acquire<MortonType>(ptNum);

		// init leaves
		Leaves leaves;
		leaves.resizePartial(ptNum);  // note: can be async
		//leaves.segOffset.resize(ptNum);

		// init interiors
		Interiors interiors;
		// note: allocation of interiors can be async
		interiors.resize(ptNum - 1);

		// calculate morton code
		parlay::parallel_for(0, ptNum,
			[&](size_t i) { BuildKernel::calcMortonCodes(i, ptNum, pts.data(), &globalBoundary, _morton.data()); }
		);

		// reorder leaves using morton code
		// note: there are multiple sorting algorithms to choose from
		parlay::integer_sort_inplace(primIdx, [&](const auto& idx) {return _morton[idx].code;});
		parlay::parallel_for(0, ptNum, [&](size_t i) {leaves.morton[i] = _morton[primIdx[i]];});

		vector<vec3f> ptsSorted(ptNum);
		parlay::parallel_for(0, ptNum, [&](size_t i) {ptsSorted[i] = pts[primIdx[i]];});

		bufferPool->release(std::move(primIdx));
		bufferPool->release(std::move(_morton));

		// calculate metrics

		parlay::parallel_for(0, ptNum - 1,
			[&](size_t i) {
				BuildKernel::calcBuildMetrics(
					i, ptNum - 1, globalBoundary, leaves.morton.data(), interiors.metrics.data(),
					interiors.splitDim.data(), interiors.splitVal.data());
			}
		);

		auto visitCount = vector<AtomicCount>(ptNum - 1);

		auto innerBuf = bufferPool->acquire<int>(ptNum - 1, 0);
		auto leafBuf = bufferPool->acquire<int>(ptNum, 0);
		BuildAid aid{ visitCount.data(), innerBuf.data(),leafBuf.data() };

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

		// calculate new indices for interiors
		auto& segLen = leafBuf;
		leaves.segOffset = parlay::scan(segLen).first;

		parlay::parallel_for(0, ptNum - 1,
			[&](size_t i) {
				BuildKernel::calcInteriorNewIdx(
					i, ptNum-1, leaves.getRawRepr(), interiors.getRawRepr(), aid.segLen,aid.leftLeafCount, interiors.mapidx.data());
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
					i, ptNum - 1, interiors.getRawRepr(),
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
					i, ptNum - 1, interiors.getRawRepr(), 
					parentSplitDim.data(), parentSplitVal.data());
			}
		);
		bufferPool->release<int>(std::move(interiors.parentSplitDim));
		bufferPool->release<mfloat>(std::move(interiors.parentSplitVal));
		interiors.parentSplitDim = std::move(parentSplitDim);
		interiors.parentSplitVal = std::move(parentSplitVal);

		// fmt::print("add {} points:\n", ptsSorted.size());
		// for (const auto& pt : ptsSorted) {
		// 	fmt::printf("(%f, %f, %f) ", pt.x, pt.y, pt.z);
		// }
		// fmt::printf("\n");
		nodeMgr->append(std::move(leaves), std::move(interiors), std::move(ptsSorted));
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

		// reset primIdx
		parlay::parallel_for(0, sizeInc, [&](uint32_t i) {return primIdx[i] = ptNum + i;});

		
		auto getMortonCode = [&](int gi) {
			if (gi >= ptNum) return mortonSorted[gi - ptNum].code;

			int iBatch, _offset;
			transformLeafIdx(gi, nodeMgrDevice.sizesAcc, nodeMgrDevice.numBatches, iBatch, _offset);
			return nodeMgrDevice.leavesBatch[iBatch].morton[_offset].code;
			};
		
		// get leafIdx
		auto leafIdxLeafSorted = parlay::remove_duplicate_integers(binIdx, maxBin + 1);
		auto leafIdxMortonSorted = parlay::integer_sort(leafIdxLeafSorted, [&](const auto& idx) {return getMortonCode(idx);});

		size_t batchLeafSize = sizeInc + leafIdxLeafSorted.size();
		auto combinedPrimIdx = bufferPool->acquire<int>(batchLeafSize);
		auto combinedBinIdx = bufferPool->acquire<int>(batchLeafSize);
		// parallel merge
		mergeZip(primIdx, leafIdxMortonSorted, binIdx, combinedPrimIdx, combinedBinIdx,
			[&](const auto& idx1, const auto& idx2) {
				return getMortonCode(idx1) < getMortonCode(idx2);
			});

		// revert removal of bins to insert
		parlay::parallel_for(0, leafIdxLeafSorted.size(), [&](size_t i) {
			UpdateKernel::revertRemoval(i, leafIdxLeafSorted.size(), leafIdxLeafSorted.data(), nodeMgrDevice);
		});

		// allocate memory for final insertion
		// note: can be async
		Leaves leaves;
		leaves.resizePartial(batchLeafSize);
		leaves.treeLocalRangeR.resize(batchLeafSize);
		leaves.derivedFrom.resize(batchLeafSize);

		Interiors interiors;
		interiors.resize(batchLeafSize-1);

		auto ptsAddFinal = bufferPool->acquire<vec3f>(batchLeafSize);
		auto treeLocalRangeL = bufferPool->acquire<int>(batchLeafSize);
		auto interiorCount = bufferPool->acquire<int>(leafIdxLeafSorted.size());
		auto interiorToLeafIdx = bufferPool->acquire<int>(batchLeafSize - leafIdxLeafSorted.size());

		auto& tempIdx = primIdx;
		tempIdx.resize(batchLeafSize);
		auto& finalPrimIdx = binIdx;
		finalPrimIdx.resize(batchLeafSize);

		parlay::parallel_for(0, batchLeafSize,
            [&](size_t i) { tempIdx[i] = i; }
		);
		// sort by combinedBinIdx
		parlay::stable_integer_sort_inplace(
			tempIdx,
			[&](const auto& idx) {return static_cast<uint32_t>(combinedBinIdx[idx]);});

		parlay::parallel_for(0, batchLeafSize,
            [&](size_t i) { finalPrimIdx[i] = combinedPrimIdx[tempIdx[i]]; }
		);
		bufferPool->release<int>(std::move(combinedPrimIdx));

		// set derivedFrom
		auto& finalBinIdx = leaves.derivedFrom;
		parlay::parallel_for(0, batchLeafSize,
			[&](size_t i) { finalBinIdx[i] = combinedBinIdx[tempIdx[i]]; }
		);
		bufferPool->release<int>(std::move(tempIdx));
		bufferPool->release<int>(std::move(combinedBinIdx));

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
		bufferPool->release<vec3f>(std::move(ptsAddSorted));

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

		// set replacedBy of leafbins
		parlay::parallel_for(0, leafIdxLeafSorted.size(), [&](size_t i) {
			int gi = leafIdxLeafSorted[i];
			int iBatch, _offset;
			transformLeafIdx(gi, nodeMgrDevice.sizesAcc, nodeMgrDevice.numBatches, iBatch, _offset);
			
			nodeMgrDevice.leavesBatch[iBatch].replacedBy[_offset] = ptNum + interiorCount[i] + i;
		});

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
			interiors.metrics.data(), interiors.splitDim.data(), interiors.splitVal.data());
		});

		//---------------------------------------------------------------------
		// build interiors
		
		auto visitCount = std::vector<AtomicCount>(batchLeafSize - 1);

		auto innerBuf = bufferPool->acquire<int>(batchLeafSize - 1, 0);
		auto leafBuf = bufferPool->acquire<int>(batchLeafSize, 0);
		BuildAid aid{ visitCount.data(), innerBuf.data(),leafBuf.data() };

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

		// calculate new indices for interiors
		auto& segLen = leafBuf;
		leaves.segOffset = parlay::scan(segLen).first;

		parlay::parallel_for(0, interiorCount.size() - 1, [&](size_t i) {
			DynamicBuildKernel::interiorMapIdxInit(i, interiorCount.size(), batchLeafSize, interiorCount.data(), interiors.mapidx.data());
		});
		bufferPool->release<int>(std::move(interiorCount));

		parlay::parallel_for(0, interiorToLeafIdx.size(),
			[&](size_t i) {
				DynamicBuildKernel::calcInteriorNewIdx(
					i, interiorToLeafIdx.size(), interiorToLeafIdx.data(),
					leaves.getRawRepr(), interiors.getRawRepr(), aid.segLen, aid.leftLeafCount, interiors.mapidx.data());
			}
		);
		bufferPool->release<int>(std::move(interiorToLeafIdx));

		// reorder interiors
		auto& rangeL = innerBuf;
		auto& rangeR = leafBuf;
		rangeR.resize(batchLeafSize - 1);
		auto splitDim = bufferPool->acquire<int>(batchLeafSize - 1);
		auto splitVal = bufferPool->acquire<mfloat>(batchLeafSize - 1);

		parlay::parallel_for(0, batchLeafSize - 1,
			[&](size_t i) {
				DynamicBuildKernel::reorderInteriors_step1(
					i, batchLeafSize - 1, interiors.getRawRepr(),
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

		auto parentSplitDim = bufferPool->acquire<int>(batchLeafSize - 1);
		auto parentSplitVal = bufferPool->acquire<mfloat>(batchLeafSize - 1);
		parlay::parallel_for(0, batchLeafSize - 1,
			[&](size_t i) {
				DynamicBuildKernel::reorderInteriors_step2(
					i, batchLeafSize - 1, interiors.getRawRepr(),
					parentSplitDim.data(), parentSplitVal.data());
			}
		);

		bufferPool->release<int>(std::move(interiors.parentSplitDim));
		bufferPool->release<mfloat>(std::move(interiors.parentSplitVal));

		interiors.parentSplitDim = std::move(parentSplitDim);
		interiors.parentSplitVal = std::move(parentSplitVal);

		// parlay::parallel_for(0, interiorCount.size(), [&](size_t i) {
		// 	DynamicBuildKernel::setSubtreeRootParentSplit(i, interiorCount.size(),
		// 	interiorCount.data(), leaves.derivedFrom.data(), nodeMgrDevice, globalBoundary,
		// 	interiors.parentSplitDim.data(), interiors.parentSplitVal.data());
		// });
		//bufferPool->release<int>(std::move(interiorCount));

		// fmt::print("splitDim:\n{}\n", interiors.splitDim);
		// fmt::print("splitVal:\n{}\n", interiors.splitVal);

		// fmt::print("add {} points:\n", ptsAddFinal.size());
		// for (const auto& pt : ptsAddFinal) {
		// 	fmt::printf("(%f, %f, %f) ", pt.x, pt.y, pt.z);
		// }
		// fmt::printf("\n");
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

		auto primIdx = bufferPool->acquire<int>(ptNum);

		// get scene boundary
		sceneBoundary.merge(reduce<AABB>(ptsAdd, MergeOp()));
		assert(globalBoundary.include(sceneBoundary));

		// calculate morton code
		parlay::parallel_for(0, sizeInc,
			[&](size_t i) { BuildKernel::calcMortonCodes(i, sizeInc, ptsAdd.data(), &globalBoundary, mortonAdd.data()); }
		);
		// assign primIdx
		parlay::parallel_for(0, sizeInc,
			[&](size_t i) { primIdxAdd[i] = i; }
		);
		// sort primIdx
		parlay::integer_sort_inplace(
			primIdxAdd,
			[&](const auto& idx) {return mortonAdd[idx].code;});

		parlay::parallel_for(0, sizeInc, [&](size_t i) { mortonAddSorted[i] = mortonAdd[primIdxAdd[i]];});
		parlay::parallel_for(0, sizeInc, [&](size_t i) { ptsAddSorted[i] = ptsAdd[primIdxAdd[i]];});
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

		vector<vec3f> ptsSortedFinal(ptNum + sizeInc);
		parlay::parallel_for(0, ptNum + sizeInc, [&](size_t i) {
			int j = primIdxFinal[i];
			int k = sizeInc;
			int t = i;
			ptsSortedFinal[i] = j < ptNum ? pts[j] : ptsAddSorted[j - ptNum];
		});

		bufferPool->release<vec3f>(std::move(ptsAddSorted));

		vector<MortonType> mortonSortedFinal(ptNum + sizeInc);
		parlay::parallel_for(0, ptNum + sizeInc, [&](size_t i) {
			int j = primIdxFinal[i];
			mortonSortedFinal[i] = j < ptNum ? leaves.morton[j] : mortonAddSorted[j - ptNum];
		});
		
		bufferPool->release<MortonType>(std::move(mortonAddSorted));

		leaves.morton = std::move(mortonSortedFinal);
		pts = std::move(ptsSortedFinal);
		ptNum = leaves.size();

		leaves.replacedBy.resize(ptNum, 0);
#ifdef ENABLE_MERKLE
		leaves.hash.resize(ptNum);
#endif

		interiors.resize(ptNum - 1);

		// calculate metrics

		parlay::parallel_for(0, ptNum - 1,
			[&](size_t i) {
				BuildKernel::calcBuildMetrics(
					i, ptNum - 1, globalBoundary, leaves.morton.data(), interiors.metrics.data(),
					interiors.splitDim.data(), interiors.splitVal.data());
			}
		);

		auto visitCount = vector<AtomicCount>(ptNum - 1);
		
		auto innerBuf = bufferPool->acquire<int>(ptNum - 1, 0);
		auto leafBuf = bufferPool->acquire<int>(ptNum, 0);
		BuildAid aid{ visitCount.data(), innerBuf.data(),leafBuf.data() };

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

		// calculate new indices for interiors
		auto& segLen = leafBuf;
		leaves.segOffset = parlay::scan(segLen).first;

		parlay::parallel_for(0, ptNum - 1,
			[&](size_t i) {
				BuildKernel::calcInteriorNewIdx(
					i, ptNum - 1, leaves.getRawRepr(), interiors.getRawRepr(), aid.segLen, aid.leftLeafCount, interiors.mapidx.data());
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
					i, ptNum - 1, interiors.getRawRepr(),
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
					i, ptNum - 1, interiors.getRawRepr(),
					parentSplitDim.data(), parentSplitVal.data());
			}
		);
		bufferPool->release<int>(std::move(interiors.parentSplitDim));
		bufferPool->release<mfloat>(std::move(interiors.parentSplitVal));

		interiors.parentSplitDim = std::move(parentSplitDim);
		interiors.parentSplitVal = std::move(parentSplitVal);
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

		for (size_t i = 0;i<hostNodeMgr.leavesBatch.size();++i) {
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
			vector<MortonType> morton(nq);

			parlay::parallel_for(0, nq,
				[&](size_t i) { BuildKernel::calcMortonCodes(i, nq, queries.data(), &globalBoundary, morton.data()); }
			);
			// note: there are multiple sorting algorithms to choose from
			parlay::integer_sort_inplace(
				responses.queryIdx, [&](const auto& idx) {return morton[idx].code;});

			queriesSorted.resize(nq);
			parlay::parallel_for(0, nq, [&](size_t i) {queriesSorted[i] = queries[responses.queryIdx[i]];});
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
						interiors.getRawRepr(), leaves.getRawRepr(), sceneBoundary, responses.exist.data());
				}
			);
		}
		else {
			parlay::parallel_for(0, nq, [&](size_t i) {
				SearchKernel::searchPoints(i,nq,target,nodeMgr->getDeviceHandle(),primSize(),sceneBoundary, responses.exist.data());
			});
		}

		return responses;
	}


	void PMKDTree::_query(const vector<RangeQuery>& queries, RangeQueryResponses& responses) const {
		size_t nq = queries.size();

		vector<RangeQuery> queriesSorted;
		const RangeQuery* target = queries.data();
		// note: sort queries as an optimization
		if (config.optimize) {
			vector<vec3f> centers(nq);
			parlay::parallel_for(0, nq,
				[&](size_t i) { centers[i] = queries[i].center(); }
			);

			vector<MortonType> morton(nq);

			parlay::parallel_for(0, nq,
				[&](size_t i) { BuildKernel::calcMortonCodes(i, nq, centers.data(), &globalBoundary, morton.data()); }
			);
			// note: there are multiple sorting algorithms to choose from
			parlay::integer_sort_inplace(
				responses.queryIdx, [&](const auto& idx) {return morton[idx].code;});

			queriesSorted.resize(nq);
			parlay::parallel_for(0, nq, [&](size_t i) {queriesSorted[i] = queries[responses.queryIdx[i]];});

			target = queriesSorted.data();
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
						sceneBoundary, responses.getRawRepr());
				}
			);
		}
		else {
			parlay::parallel_for(0, nq, [&](size_t i) {
				SearchKernel::searchRanges(i, nq, target,
				nodeMgr->getDeviceHandle(), primSize(), sceneBoundary,
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

	void PMKDTree::insert(const vector<vec3f>& ptsAdd) {
		if (ptsAdd.empty()) return;

		isStatic = false;
		buildIncrement(ptsAdd);
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
			auto primIdx = bufferPool->acquire<int>(nq);
			auto morton = bufferPool->acquire<MortonType>(nq);
			ptsRemoveSorted = bufferPool->acquire<vec3f>(nq);

			parlay::parallel_for(0, nq,
				[&](size_t i) { primIdx[i] = i; }
            );

			parlay::parallel_for(0, nq,
				[&](size_t i) { BuildKernel::calcMortonCodes(i, nq, ptsRemove.data(), &globalBoundary, morton.data()); }
			);
			// note: there are multiple sorting algorithms to choose from
			parlay::integer_sort_inplace(
				primIdx, [&](const auto& idx) {return morton[idx].code;});
			
			parlay::parallel_for(0, nq, [&](size_t i) {ptsRemoveSorted[i] = ptsRemove[primIdx[i]]; });

			bufferPool->release(std::move(primIdx));
			bufferPool->release(std::move(morton));

			target = ptsRemoveSorted.data();
		}

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

			parlay::parallel_for(0, nq, [&](size_t i) {
				UpdateKernel::removePoints_step2(i, nq, primSize(), binIdx.data(), leaves.getRawRepr(), interiors.getRawRepr());
			});
			// fmt::print("segOffset: {}\n{}\n", leaves.segOffset.size(), leaves.segOffset);
			// fmt::print("alterState: {}\n", interiors.alterState.size());
			// for (size_t i = 0;i < interiors.size();i++) {
			// 	std::bitset<8> binary(interiors.alterState[i].cnt.load(std::memory_order_relaxed));
			// 	fmt::print("{}, ", binary.to_string());
			// }
			// fmt::print("\n");
		}
		else {
			auto nodeMgrDeviceHandle = nodeMgr->getDeviceHandle();
			
			parlay::parallel_for(0, nq, [&](size_t i) {
				UpdateKernel::removePoints_step1(i, nq, target, nodeMgrDeviceHandle, primSize(), binIdx.data());
				}
			);

			parlay::parallel_for(0, nq, [&](size_t i) {
                UpdateKernel::removePoints_step2(i, nq, binIdx.data(), nodeMgrDeviceHandle);
			});
		}
		if (!ptsRemoveSorted.empty()) bufferPool->release(std::move(ptsRemoveSorted));
		bufferPool->release(std::move(binIdx));
	}
}