#include <tree/device_helper.h>
#include <tree/kernel.h>

namespace pmkd {
	void BuildKernel::reduceBoundary(int idx, int size, INPUT(vec3f*) pts, OUTPUT(AABB*) boundary) {

	}

	void BuildKernel::calcMortonCodes(int idx, int size, INPUT(vec3f*) pts, INPUT(AABB*) gboundary,
		OUTPUT(MortonType*) morton) {
		
		if (idx >= size) return;
		vec3f offset = (pts[idx] - gboundary->ptMin);
		offset.x /= (gboundary->ptMax.x - gboundary->ptMin.x);
		offset.y /= (gboundary->ptMax.y - gboundary->ptMin.y);
		offset.z /= (gboundary->ptMax.z - gboundary->ptMin.z);

		morton[idx] = MortonType::calculate(offset.x, offset.y, offset.z);
	}
	
	void BuildKernel::calcBuildMetrics(int idx, int interiorSize, const AABB& gBoundary, INPUT(MortonType*) morton,
		OUTPUT(uint8_t*) metrics, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal) {
		if (idx >= interiorSize) return;

		// reflects the highest differing bit between the keys covered by interior node idx
		uint8_t metric = MortonType::calcMetric(morton[idx], morton[idx + 1]);
		metrics[idx] = metric;
		MortonType::calcSplit(metric, morton[idx + 1], gBoundary.ptMin, gBoundary.ptMax,
			splitDim + idx, splitVal + idx);
	}

	void BuildKernel::buildInteriors(int idx, int leafSize, const LeavesRawRepr leaves,
		InteriorsRawRepr interiors, BuildAid aid) {
		
		if (idx >= leafSize) return;

		int parent;
		bool isRC = idx != 0 && (idx == leafSize - 1 || aid.metrics[idx - 1] <= aid.metrics[idx]);
		if (!isRC) {
			// is left child of Interior idx
			parent = idx;
			//leaves.parentSplitDim[idx] = interiors.splitDim[parent];
			//leaves.parentSplitVal[idx] = interiors.splitVal[parent];
			//interiors.lc[parent] = toLeafIdx(idx);
			interiors.rangeL[parent] = idx;
			aid.segLen[idx] = 1;
			aid.leftLeafCount[idx] = 1;
		}
		else {
			// is right child of Interior idx-1
			parent = idx - 1;
			//leaves.parentSplitDim[idx] = interiors.splitDim[parent];
			//leaves.parentSplitVal[idx] = interiors.splitVal[parent];
			//interiors.rc[parent] = toLeafIdx(idx);
			interiors.rangeR[parent] = idx;
		}
		leaves.parent[idx] = encodeParentCode(parent, isRC);

		// choose parent in bottom-up fashion. O(n)
		int current, left, right;
		while (setVisitCountBottomUp(aid.visitCount[parent], isRC)) {
			current = parent;

			left = interiors.rangeL[current];
			right = interiors.rangeR[current];

			if (left == 0 && right == leafSize - 1) {
				//root
				interiors.parent[current] = -1;
				break;
			}

			isRC = left != 0 && (right == leafSize - 1 || aid.metrics[left - 1] <= aid.metrics[right]);
			if (!isRC) {
				// is left child of Interior right
				parent = right;
				//interiors.parent[current] = parent;
				//interiors.lc[parent] = toInteriorIdx(current);
				interiors.rangeL[parent] = left;
				//printf("interior %d has rangeL %d\n", parent, left);

				//aid.leftLeafCount[parent] = old;
				aid.segLen[left] ++;  // add LCL value of the leftmost leaf
				aid.leftLeafCount[parent] = aid.segLen[left]; // count the order of Interior parent
			}
			else {
				// is right child of Interior left-1
				parent = left - 1;
				//interiors.parent[current] = parent;
				//interiors.rc[parent] = toInteriorIdx(current);
				interiors.rangeR[parent] = right;
			}
			interiors.parent[current] = encodeParentCode(parent, isRC);
		}
	}

	// optimized version of buildInteriors by removing branches
	void BuildKernel::buildInteriors_opt(int idx, int leafSize, const LeavesRawRepr leaves, INPUT(int*) metrics,
		OUTPUT(int*) range[2], OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal,
		OUTPUT(int*) interiorParent, BuildAid aid) {
		
		if (idx >= leafSize) return;

		int isRC = idx != 0 && (idx == leafSize - 1 || metrics[idx - 1] <= metrics[idx]); // 1 if is right child
		int parent = idx - isRC;
		leaves.parent[idx] = encodeParentCode(parent, isRC);

		aid.segLen[idx] = 1 - isRC;
		aid.leftLeafCount[idx] = 1 - isRC;

		// note: this may not be a useful optimization
		// rangeL or rangeR
		int* rangePtr = range[isRC];
		rangePtr[parent] = idx;

		// choose parent in bottom-up fashion. O(n)
		int current;
		while (setVisitCountBottomUp(aid.visitCount[parent], isRC)) {
			current = parent;

			int LR[2] = { range[0][current] ,range[1][current] };  // left, right
			const auto& left = LR[0];
			const auto& right = LR[1];

			if (left == 0 && right == leafSize - 1) {
				//root
				interiorParent[current] = -1;
				break;
			}
			isRC = left != 0 && (right == leafSize - 1 || metrics[left - 1] <= metrics[right]); // 1 if is right child
			parent = LR[1 - isRC] - isRC;
			range[isRC][parent] = LR[isRC];
			if (!isRC) {
				aid.segLen[left]++;  // add LCL value of the leftmost leaf
				aid.leftLeafCount[parent] = aid.segLen[left]; // count the order of Interior parent
			}

			interiorParent[current] = encodeParentCode(parent, isRC);
		}
	}

	void BuildKernel::calcInteriorNewIdx(int idx, int size, const LeavesRawRepr leaves, const InteriorsRawRepr interiors,
		INPUT(int*) segLen, INPUT(int*) leftLeafCount, OUTPUT(int*) mapidx) {
		
		if (idx >= size) return;

		int binIdx = interiors.rangeL[idx];
		int idx_new = leaves.segOffset[binIdx] + segLen[binIdx] - leftLeafCount[idx];
		mapidx[idx] = idx_new;
	}

	// in place
	void BuildKernel::reorderInteriors(int idx, int interiorSize, INPUT(int*) mapidx, const InteriorsRawRepr interiors,
		OUTPUT(int*) rangeL, OUTPUT(int*) rangeR, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal, OUTPUT(int*) parent) {

		if (idx >= interiorSize) return;
		int mapped_idx = mapidx[idx];

		rangeL[mapped_idx] = interiors.rangeL[idx];
		rangeR[mapped_idx] = interiors.rangeR[idx];

		splitDim[mapped_idx] = interiors.splitDim[idx];
		splitVal[mapped_idx] = interiors.splitVal[idx];

		int parentIdx;
		bool isRC;
		decodeParentCode(interiors.parent[idx], parentIdx, isRC);
		parent[mapped_idx] = parentIdx >= 0 ? encodeParentCode(mapidx[parentIdx], isRC) : -1;
	}

	void BuildKernel::remapLeafParents(int idx, int leafSize, INPUT(int*) mapidx, LeavesRawRepr leaves) {

		if (idx >= leafSize) return;

		int parentIdx;
		bool isRC;
		decodeParentCode(leaves.parent[idx], parentIdx, isRC);
		int mapped_idx = mapidx[parentIdx];
		leaves.parent[idx] = encodeParentCode(mapped_idx, isRC);
	}

#ifdef ENABLE_MERKLE
	void BuildKernel::calcLeafHash(int idx, int size, INPUT(vec3f*) pts, INPUT(int*) removeFlag, OUTPUT(hash_t*) leafHash) {
		if (idx >= size) return;

		computeDigest(leafHash + idx, pts[idx].x, pts[idx].y, pts[idx].z, removeFlag[idx] == -1);
	}

	void BuildKernel::calcInteriorHash(int idx, int leafSize, const LeavesRawRepr leaves,
		InteriorsRawRepr interiors, OUTPUT(AtomicCount*) visitCount) {
		if (idx >= leafSize) return;

		bool isRC = idx != 0 && (idx == leafSize - 1 || interiors.metrics[idx - 1] <= interiors.metrics[idx]);
		int parent = interiors.mapidx[idx - isRC];
		int current = idx;
		const hash_t* childHash = leaves.hash + current;

		// choose parent in bottom-up fashion. O(n)
		while (setVisitCountBottomUp(visitCount[parent], isRC))
		{
			current = parent;

			int LR[2] = { interiors.rangeL[current] ,interiors.rangeR[current] };  // left, right
			const auto& left = LR[0];
			const auto& right = LR[1];

			// if (isRC) {  // get left child hash
			// 	if (current < R - 1) otherChildHash = &interiors.hash[current + 1];  // left is interior
			// 	else otherChildHash = &leaves.hash[left];                            // left is leaf
			// }
			// else {       // get right child hash
			// 	if (current < R - 1) {
			// 		int nextBin = interiors.rangeR[current + 1] + 1;
			// 		int nextIdx = leaves.segOffset[nextBin];
			// 		if (nextBin == leafSize - 1 || nextIdx == leaves.segOffset[nextBin + 1])
			// 			otherChildHash = &leaves.hash[nextBin];
			// 		else otherChildHash = &interiors.hash[nextIdx];
			// 	}
			// 	else {
			// 		otherChildHash = &leaves.hash[left + 1];
			// 	}
			// }
			const hash_t* otherChildHash;
			getOtherChildHash(leaves, interiors, left, current, leafSize, isRC, otherChildHash);

			computeDigest(interiors.hash + current, childHash, otherChildHash,
				interiors.splitDim[current], interiors.splitVal[current]);

			if (current == 0) break; // root

			childHash = interiors.hash + current;

			isRC = left != 0 && (right == leafSize - 1 || interiors.metrics[left - 1] <= interiors.metrics[right]);
			parent = interiors.mapidx[LR[1 - isRC] - isRC];
		}
	}
#endif
}