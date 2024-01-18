#include <tree/kernel.h>
#include <common/util/atomic_ext.h>

namespace pmkd {
	void SearchKernel::searchPoints(int qIdx, int qSize, const Query* qPts, const vec3f* pts, int leafSize,
		InteriorsRawRepr interiors, LeavesRawRepr leaves, const AABB& boundary, QueryResponse* resp) {
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
				resp[qIdx].exist = pts[begin] == pt;
				break;
			}
		}
	}

	void SearchKernel::searchPoints_opt(int qIdx, int qSize, const Query* qPts, const vec3f* pts, int leafSize,
		InteriorsRawRepr interiors, LeavesRawRepr leaves, const AABB& boundary,
		int rangeL, std::atomic<int>& rangeR, QueryResponse* resp) {
		if (qIdx >= qSize) return;
		const vec3f& pt = qPts[qIdx];
		if (!boundary.include(pt)) return;

		// do sprouting
		int L = 0, R = 0;
		bool onRight;
		int interiorIdx = 0;
		for (int begin = rangeL; begin < leafSize; begin++) {
			L = leaves.segOffset[begin];
			R = begin == leafSize - 1 ? L : leaves.segOffset[begin + 1];
			onRight = false;
			for (interiorIdx = L; interiorIdx < R; interiorIdx++) {
				int splitDim = interiors.splitDim[interiorIdx];
				mfloat splitVal = interiors.splitVal[interiorIdx];
				onRight = pt[splitDim] >= splitVal;
				if (onRight) {
					// goto right child
					//begin = interiorIdx < R - 1 ? 
					//	interiors.rangeR[interiorIdx + 1] + 1 : interiors.rangeR[interiorIdx];
					//begin--;
					begin = interiorIdx < R - 1 ? interiors.rangeR[interiorIdx + 1] : begin;
					break;
				}
			}
			if (!onRight) {
				// hit leaf with index <begin>
				//resp[qIdx].exist = pts[leaves.primIdx[begin]] == pt;
				resp[qIdx].exist = pts[begin] == pt;
				// atomic max
				atomic_fetch_max_explicit(&rangeR, begin, std::memory_order_release,
					[](int a, int b) {return std::max(a, b);});
			}
		}
	}

	void SearchKernel::searchRanges(int qIdx, int qSize, const RangeQuery* qRanges, const vec3f* pts, int leafSize,
		InteriorsRawRepr interiors, LeavesRawRepr leaves, const AABB& boundary,
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
			if (L < R && L > 0) {
				splitDim = interiors.parentSplitDim[L];
				splitVal = interiors.parentSplitVal[L];
				if (box.ptMax[splitDim] < splitVal) continue;
			}

			for (interiorIdx = L; interiorIdx < R; interiorIdx++) {
				splitDim = interiors.splitDim[interiorIdx];
				splitVal = interiors.splitVal[interiorIdx];

				onRight = box.ptMin[splitDim] >= splitVal;
				if (onRight) {
					// goto right child
					//begin = interiorIdx < R - 1 ? 
					//	interiors.rangeR[interiorIdx + 1] + 1 : interiors.rangeR[interiorIdx];
					//begin--;
					begin = interiorIdx < R - 1 ? interiors.rangeR[interiorIdx + 1] : begin;
					break;
				}
			}
			if (onRight) continue;

			// hit leaf with index <begin>
			//auto& target = pts[leaves.primIdx[begin]];
			auto& target = pts[begin];
			if (box.include(target)) {
				auto& respSize = *(resps.getSizePtr(qIdx));
				resps.getBufPtr(qIdx)[respSize++] = target;  // note: check correctness on GPU
				if (respSize >= resps.capPerResponse) break;
			}
		}
	}
}