#include <tree/device_helper.h>
#include <tree/kernel.h>

namespace pmkd {

	void SearchKernel::searchPoints(int qIdx, int qSize, const Query* qPts, const vec3f* pts, int leafSize,
		const InteriorsRawRepr interiors, const LeavesRawRepr leaves, const AABB& boundary, uint8_t* exist) {
		if (qIdx >= qSize) return;
		const vec3f& pt = qPts[qIdx];
		if (!boundary.include(pt)) return;

		// do sprouting
		int L = 0, R = 0;
		bool onRight;
		int interiorIdx = 0;
		for (int begin = 0; begin < leafSize; begin++) {
			L = leaves.segOffset[begin];
			R = begin == leafSize - 1 ? L : leaves.segOffset[begin + 1];
			onRight = false;
			for (interiorIdx = L; interiorIdx < R; interiorIdx++) {
				if (isInteriorRemoved(interiors.removeState[interiorIdx])) {
					return;
				}

				int splitDim = interiors.splitDim[interiorIdx];
				mfloat splitVal = interiors.splitVal[interiorIdx];
				onRight = pt[splitDim] >= splitVal;
				if (onRight) {
					// goto right child
					//begin = interiorIdx < R - 1 ? 
					//	interiors.rangeR[interiorIdx + 1] + 1 : begin+1;
					//begin--;
					begin = interiorIdx < R - 1 ? interiors.rangeR[interiorIdx + 1] : begin;
					break;
				}
			}
			if (!onRight) {
				// hit leaf with index <begin>
				//resp[qIdx].exist = pts[leaves.primIdx[begin]] == pt;
				exist[qIdx] = leaves.replacedBy[begin] == 0 && pts[begin] == pt;
				break;
			}
		}
	}

	void SearchKernel::searchPoints(int qIdx, int qSize, const Query* qPts, const NodeMgrDevice nodeMgr, int totalLeafSize,
		const AABB& boundary, uint8_t* exist) {
		if (qIdx >= qSize) return;
		const vec3f& pt = qPts[qIdx];
		if (!boundary.include(pt)) return;

		// do sprouting
		int iBatch = 0, localLeafIdx = 0;
		int mainTreeLeafSize = nodeMgr.sizesAcc[0];
		int L = 0, R = 0;
		bool onRight;
		int interiorIdx = 0;


		int globalLeafIdx = 0;
		while (globalLeafIdx < totalLeafSize) {
			transformLeafIdx(globalLeafIdx, nodeMgr.sizesAcc, nodeMgr.numBatches, iBatch, localLeafIdx);

			const auto& leaves = nodeMgr.leavesBatch[iBatch];
			const auto& interiors = nodeMgr.interiorsBatch[iBatch];
			//const auto& treeLocalRangeR = nodeMgr.treeLocalRangeR[iBatch];

			int rBound = iBatch == 0 ? mainTreeLeafSize : leaves.treeLocalRangeR[localLeafIdx];

 			while (localLeafIdx < rBound) {
				int oldLocalLeafIdx = localLeafIdx;
				
				L = leaves.segOffset[localLeafIdx];
				R = localLeafIdx == rBound - 1 ? L : leaves.segOffset[localLeafIdx + 1];
				onRight = false;
				for (interiorIdx = L; interiorIdx < R; interiorIdx++) {
					if (isInteriorRemoved(interiors.removeState[interiorIdx]))
					 	return;
					
					int splitDim = interiors.splitDim[interiorIdx];
					mfloat splitVal = interiors.splitVal[interiorIdx];
					onRight = pt[splitDim] >= splitVal;
					if (onRight) {
						// goto right child
						localLeafIdx = interiorIdx < R - 1 ? interiors.rangeR[interiorIdx + 1] : localLeafIdx;
						break;
					}
				}
				if (!onRight) {
					int globalSubstitute = leaves.replacedBy[localLeafIdx];
					if (globalSubstitute <= 0) { // this leaf is valid or removed (i.e. not replaced)
						exist[qIdx] = !(globalSubstitute < 0) && nodeMgr.ptsBatch[iBatch][localLeafIdx] == pt;
						return;
					}
					// leaf is replaced
					globalLeafIdx = globalSubstitute;
					break;
				}
				++localLeafIdx;
				globalLeafIdx += localLeafIdx - oldLocalLeafIdx;
			}
		}
	}

	void SearchKernel::searchRanges(int qIdx, int qSize, const RangeQuery* qRanges, const vec3f* pts, int leafSize,
		const InteriorsRawRepr interiors, const LeavesRawRepr leaves, const AABB& boundary,
		RangeQueryResponsesRawRepr resps) {
		
		if (qIdx >= qSize) return;
		const AABB& box = qRanges[qIdx];
		if (!boundary.overlap(box)) return;

		// do sprouting
		int L = 0, R = 0;
		bool onRight;
		int interiorIdx = 0;
		int splitDim;
		mfloat splitVal;
		for (int begin = 0; begin < leafSize; begin++) {
			L = leaves.segOffset[begin];
			R = begin == leafSize - 1 ? L : leaves.segOffset[begin + 1];
			onRight = false;

			// skip if box does not overlap the subtree rooted at L
			if (L > 0 && L < R) {
				int parent;
				bool isRC;
				decodeParentCode(interiors.parent[L], parent, isRC);
				splitDim = interiors.splitDim[parent];
				splitVal = interiors.splitVal[parent];
				if (box.ptMax[splitDim] < splitVal) {
					begin = interiors.rangeR[L];
					continue;
				}
			}

			for (interiorIdx = L; interiorIdx < R; interiorIdx++) {
				bool isRemoved = isInteriorRemoved(interiors.removeState[interiorIdx]); // removed
				if (!isRemoved) {
					splitDim = interiors.splitDim[interiorIdx];
					splitVal = interiors.splitVal[interiorIdx];

					onRight = box.ptMin[splitDim] >= splitVal;
				}
				onRight = onRight || isRemoved;

				if (onRight) {
					//begin = interiorIdx < R - 1 && !isRemoved ? interiors.rangeR[interiorIdx + 1] : begin + isRemoved;
					if (interiorIdx < R - 1 || isRemoved) {
						begin = interiors.rangeR[interiorIdx + (1 - isRemoved)];
					}
					break;
				}
			}
			if (onRight) continue;

			// hit leaf with index <begin>
			if (leaves.replacedBy[begin] < 0) continue;  // leaf is removed

			const auto& target = pts[begin];
			if (box.include(target)) {
				auto& respSize = *(resps.getSizePtr(qIdx));
				resps.getBufPtr(qIdx)[respSize++] = target;  // note: check correctness on GPU
				if (respSize >= resps.capPerResponse) break;
			}
		}
	}

	void SearchKernel::searchRanges(int qIdx, int qSize, const RangeQuery* qRanges, const NodeMgrDevice nodeMgr, int totalLeafSize,
		const AABB& boundary, RangeQueryResponsesRawRepr resps) {
		
		if (qIdx >= qSize) return;
		const AABB& box = qRanges[qIdx];
		if (!boundary.overlap(box)) return;

		// do sprouting
		int iBatch = 0, localLeafIdx = 0;
		int mainTreeLeafSize = nodeMgr.sizesAcc[0];
		
		int L = 0, R = 0;
		bool onRight;
		int interiorIdx = 0;
		int splitDim;
		mfloat splitVal;

		int globalLeafIdx = 0;
		int state = 0;   // 0: init, 1: deeper, 2: stack return

		while (globalLeafIdx < totalLeafSize) {
			transformLeafIdx(globalLeafIdx, nodeMgr.sizesAcc, nodeMgr.numBatches, iBatch, localLeafIdx);

			const auto& leaves = nodeMgr.leavesBatch[iBatch];
			const auto& interiors = nodeMgr.interiorsBatch[iBatch];
			//const auto& treeLocalRangeR = nodeMgr.treeLocalRangeR[iBatch];

			int rBound = iBatch == 0 ? mainTreeLeafSize : leaves.treeLocalRangeR[localLeafIdx];
			if (state == 2) {
				globalLeafIdx++;
				localLeafIdx++;
			}
			state = 2;

			int oldLocalLeafIdx = localLeafIdx;
			for (;localLeafIdx < rBound;globalLeafIdx += ++localLeafIdx - oldLocalLeafIdx) {
				oldLocalLeafIdx = localLeafIdx;

				L = leaves.segOffset[localLeafIdx];
				R = localLeafIdx == rBound - 1 ? L : leaves.segOffset[localLeafIdx + 1];

				

				onRight = false;
				
				//if (L < R) {
				int parentCode = L < R ? interiors.parent[L] : leaves.parent[localLeafIdx];
					if (parentCode >= 0) {
						int parent;
						bool isRC;
						decodeParentCode(parentCode, parent, isRC);
						splitDim = interiors.splitDim[parent];
						splitVal = interiors.splitVal[parent];
						if (box.ptMax[splitDim] < splitVal) {
							if (L < R) localLeafIdx = interiors.rangeR[L];
							continue;
						}
					}
				//}

				for (interiorIdx = L; interiorIdx < R; interiorIdx++) {
					bool isRemoved = isInteriorRemoved(interiors.removeState[interiorIdx]);   // note: judging removaL may not increase performance
					if (!isRemoved) {
						splitDim = interiors.splitDim[interiorIdx];
						splitVal = interiors.splitVal[interiorIdx];
						onRight = box.ptMin[splitDim] >= splitVal;
					}
					onRight = onRight || isRemoved;

					if (onRight) {
						//localLeafIdx = interiorIdx < R - 1 && !isRemoved ? interiors.rangeR[interiorIdx + 1] : localLeafIdx + isRemoved;
						if (interiorIdx < R - 1 || isRemoved) {
							localLeafIdx = interiors.rangeR[interiorIdx + (1 - isRemoved)];
						}
						break;
					}
				}
				if (!onRight) {
					int globalSubstitute = leaves.replacedBy[localLeafIdx];
					if (globalSubstitute == 0) { // this leaf is valid (i.e. not replaced or removed)
						const auto& target = nodeMgr.ptsBatch[iBatch][localLeafIdx];
						if (box.include(target)) {
							auto& respSize = *(resps.getSizePtr(qIdx));
							resps.getBufPtr(qIdx)[respSize++] = target;  // note: check correctness on GPU
							if (respSize >= resps.capPerResponse) return;
						}
					}
					else if (globalSubstitute > 0) {
						// leaf is replaced
						globalLeafIdx = globalSubstitute;
						state = 1;
						break;
					}
				}
			}
			if (state == 2) {
				// equal to stack return
				globalLeafIdx = iBatch > 0 ? leaves.derivedFrom[rBound - 1] : totalLeafSize;
			}
		}
	}

#ifdef ENABLE_MERKLE
	void SearchKernel::searchRangesVerifiable_step1(int qIdx, int qSize, const RangeQuery* qRanges, const NodeMgrDevice nodeMgr, int totalLeafSize,
		OUTPUT(int*) fCnt, OUTPUT(int*) mCnt, OUTPUT(int*) hCnt) {
		
		if (qIdx >= qSize) return;
		const AABB& box = qRanges[qIdx];

		int fc = 0, mc = 0, hc = 0;
		// do sprouting
		int iBatch = 0, localLeafIdx = 0;
		int mainTreeLeafSize = nodeMgr.sizesAcc[0];

		int L = 0, R = 0;
		bool onRight;
		int interiorIdx = 0;
		int splitDim;
		mfloat splitVal;

		int globalLeafIdx = 0;
		int state = 0;   // 0: init, 1: deeper, 2: stack return

		while (globalLeafIdx < totalLeafSize) {
			transformLeafIdx(globalLeafIdx, nodeMgr.sizesAcc, nodeMgr.numBatches, iBatch, localLeafIdx);

			const auto& leaves = nodeMgr.leavesBatch[iBatch];
			const auto& interiors = nodeMgr.interiorsBatch[iBatch];
			//const auto& treeLocalRangeR = nodeMgr.treeLocalRangeR[iBatch];

			int rBound = iBatch == 0 ? mainTreeLeafSize : leaves.treeLocalRangeR[localLeafIdx];
			if (state == 2) {
				globalLeafIdx++;
				localLeafIdx++;
			}
			state = 2;

			int oldLocalLeafIdx = localLeafIdx;
			for (;localLeafIdx < rBound;globalLeafIdx += ++localLeafIdx - oldLocalLeafIdx) {
				oldLocalLeafIdx = localLeafIdx;

				L = leaves.segOffset[localLeafIdx];
				R = localLeafIdx == rBound - 1 ? L : leaves.segOffset[localLeafIdx + 1];


				onRight = false;

				int parentCode = L < R ? interiors.parent[L] : leaves.parent[localLeafIdx];
				if (parentCode >= 0) {
					int parent;
					bool isRC;
					decodeParentCode(parentCode, parent, isRC);
					splitDim = interiors.splitDim[parent];
					splitVal = interiors.splitVal[parent];
					if (box.ptMax[splitDim] < splitVal) {
						if (L < R) localLeafIdx = interiors.rangeR[L];
						hc++;
						continue;
					}
				}

				for (interiorIdx = L; interiorIdx < R; interiorIdx++) {
					bool isRemoved = isInteriorRemoved(interiors.removeState[interiorIdx]);   // note: judging removaL may not increase performance
					if (!isRemoved) {
						splitDim = interiors.splitDim[interiorIdx];
						splitVal = interiors.splitVal[interiorIdx];
						onRight = box.ptMin[splitDim] >= splitVal;

						mc++;
					}
					onRight = onRight || isRemoved;

					if (onRight) {
						// goto right child
						//localLeafIdx = interiorIdx < R - 1 && !isRemoved ? interiors.rangeR[interiorIdx + 1] : localLeafIdx + isRemoved;
						if (interiorIdx < R - 1 || isRemoved) {
							localLeafIdx = interiors.rangeR[interiorIdx + (1 - isRemoved)];
						}
						hc++;
						break;
					}
				}
				if (!onRight) {
					int globalSubstitute = leaves.replacedBy[localLeafIdx];
					if (globalSubstitute <= 0) { // this leaf is not replaced
						fc++;
					}
					else {
						// leaf is replaced
						globalLeafIdx = globalSubstitute;
						state = 1;
						break;
					}
				}
			}
			if (state == 2) {
				// equal to stack return
				globalLeafIdx = iBatch > 0 ? leaves.derivedFrom[rBound - 1] : totalLeafSize;
			}
		}
		fCnt[qIdx] = fc;
		mCnt[qIdx] = mc;
		hCnt[qIdx] = hc;
	}

	void SearchKernel::searchRangesVerifiable_step2(int qIdx, int qSize, const RangeQuery* qRanges, const NodeMgrDevice nodeMgr, int totalLeafSize,
		INPUT(int*) fOffset, INPUT(int*) mOffset, INPUT(int*) hOffset,
		FNodesRawRepr fNodes, MNodesRawRepr mNodes, HNodesRawRepr hNodes) {
		if (qIdx >= qSize) return;
		const AABB& box = qRanges[qIdx];
		uint32_t fi = fOffset[qIdx], mi = mOffset[qIdx], hi = hOffset[qIdx];

		// do sprouting
		int iBatch = 0, localLeafIdx = 0;
		int mainTreeLeafSize = nodeMgr.sizesAcc[0];

		int L = 0, R = 0;
		bool onRight;
		int interiorIdx = 0;
		int splitDim;
		mfloat splitVal;

		int globalLeafIdx = 0;
		int state = 0;   // 0: init, 1: deeper, 2: stack return

		while (globalLeafIdx < totalLeafSize) {
			transformLeafIdx(globalLeafIdx, nodeMgr.sizesAcc, nodeMgr.numBatches, iBatch, localLeafIdx);
			uint32_t globalOffset = globalLeafIdx - localLeafIdx;

			const auto& leaves = nodeMgr.leavesBatch[iBatch];
			const auto& interiors = nodeMgr.interiorsBatch[iBatch];
			//const auto& treeLocalRangeR = nodeMgr.treeLocalRangeR[iBatch];

			int rBound = iBatch == 0 ? mainTreeLeafSize : leaves.treeLocalRangeR[localLeafIdx];
			if (state == 2) {
				globalLeafIdx++;
				localLeafIdx++;
			}
			state = 2;

			int oldLocalLeafIdx = localLeafIdx;
			for (;localLeafIdx < rBound;globalLeafIdx += ++localLeafIdx - oldLocalLeafIdx) {
				oldLocalLeafIdx = localLeafIdx;

				L = leaves.segOffset[localLeafIdx];
				R = localLeafIdx == rBound - 1 ? L : leaves.segOffset[localLeafIdx + 1];


				onRight = false;

				int parentCode = L < R ? interiors.parent[L] : leaves.parent[localLeafIdx];
				if (parentCode >= 0) {
					int parent;
					bool isRC;
					decodeParentCode(parentCode, parent, isRC);
					splitDim = interiors.splitDim[parent];
					splitVal = interiors.splitVal[parent];
					if (box.ptMax[splitDim] < splitVal) {
						hash_t* _hash;
						if (L < R) {
							_hash = interiors.hash + L;
							localLeafIdx = interiors.rangeR[L];
						}
						else {
                            _hash = leaves.hash + localLeafIdx;
						}
						hNodes.fromNode(_hash, parentCode, globalOffset, iBatch, hi++);
						continue;
					}
				}

				for (interiorIdx = L; interiorIdx < R; interiorIdx++) {
					bool isRemoved = isInteriorRemoved(interiors.removeState[interiorIdx]);
					if (!isRemoved) {
						splitDim = interiors.splitDim[interiorIdx];
						splitVal = interiors.splitVal[interiorIdx];
						onRight = box.ptMin[splitDim] >= splitVal;

						mNodes.fromInterior(interiors, mi++, interiorIdx, globalOffset, iBatch);
					}

					onRight = onRight || isRemoved;

					if (onRight) {
						uint32_t ni;
						int _parentCode;
						hash_t* _hash;
						//localLeafIdx = interiorIdx < R - 1 && !isRemoved ? interiors.rangeR[interiorIdx + 1] : localLeafIdx + isRemoved;
						if (interiorIdx < R - 1 || isRemoved) {
							ni = interiorIdx + (1 - isRemoved);
							_parentCode = interiors.parent[ni];
							_hash = interiors.hash + ni;

							localLeafIdx = interiors.rangeR[ni];
						}
						else {
							ni = localLeafIdx;
							_parentCode = leaves.parent[ni];
							_hash = leaves.hash + ni;
						}
						hNodes.fromNode(_hash, _parentCode, globalOffset, iBatch, hi++);
						break;
					}
				}
				if (!onRight) {
					int globalSubstitute = leaves.replacedBy[localLeafIdx];
					if (globalSubstitute <= 0) { // this leaf is not replaced
						fNodes.fromLeaf(leaves, nodeMgr.ptsBatch[iBatch], fi++, localLeafIdx, globalOffset);
					}
					else {
						// leaf is replaced
						globalLeafIdx = globalSubstitute;
						state = 1;
						break;
					}
				}
			}
			if (state == 2) {
				// equal to stack return
				globalLeafIdx = iBatch > 0 ? leaves.derivedFrom[rBound - 1] : totalLeafSize;
			}
		}
	}
#endif
}