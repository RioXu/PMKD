#pragma once
#include <vector>
#include <atomic>
#include "node.h"
#include "query_response.h"

namespace pmkd {
#define INPUT(ptr_t) const ptr_t __restrict__
#define OUTPUT(ptr_t) ptr_t __restrict__

	struct BuildKernel {
		static void reduceBoundary(int idx, int size, INPUT(vec3f*) pts, OUTPUT(AABB*) boundary);

		static void calcMortonCodes(int idx, int size, INPUT(vec3f*) pts, INPUT(AABB*) gboundary,
			OUTPUT(MortonType*) morton);

		//static void reorderLeaves(int idx, int size, LeavesRawRepr&& leaves, LeavesRawRepr&& leaves_sorted, int* mapidx);

		static void calcBuildMetrics(int idx, int interiorSize, const AABB& gBoundary, LeavesRawRepr leaves,
			OUTPUT(int*) metrics, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal);

		static void buildInteriors(int idx, int leafSize, const LeavesRawRepr leaves,
			InteriorsRawRepr interiors, BuildAid aid);

		// optimized version of buildInteriors by removing branches
		static void buildInteriors_opt(int idx, int leafSize, const LeavesRawRepr leaves,
			OUTPUT(int*) range[2], OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal,
			OUTPUT(int*) parentSplitDim, OUTPUT(mfloat*) parentSplitVal,
			BuildAid aid);

		static void calcInteriorNewIdx(int idx, int size, const LeavesRawRepr leaves, const InteriorsRawRepr interiors,
			INPUT(int*) segLen, INPUT(int*) leftLeafCount, OUTPUT(int*) mapidx);

		// in place
		static void reorderInteriors_step1(int idx, int interiorSize, const InteriorsRawRepr interiors, INPUT(int*) mapidx,
			OUTPUT(int*) rangeL, OUTPUT(int*) rangeR, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal);

		static void reorderInteriors_step2(int idx, int interiorSize, const InteriorsRawRepr interiors, INPUT(int*) mapidx,
			OUTPUT(int*) parentSplitDim, OUTPUT(mfloat*) parentSplitVal);
	};



	struct SearchKernel {
		static void searchPoints(int qIdx, int qSize, const Query* qPts, const vec3f* pts, int leafSize,
			InteriorsRawRepr interiors, LeavesRawRepr leaves, const AABB& boundary, QueryResponse* resp);

		static void searchPoints_opt(int qIdx, int qSize, const Query* qPts, const vec3f* pts, int leafSize,
			InteriorsRawRepr interiors, LeavesRawRepr leaves, const AABB& boundary,
			int rangeL, std::atomic<int>& rangeR, QueryResponse* resp);

		static void searchRanges(int qIdx, int qSize, const RangeQuery* qRanges, const vec3f* pts, int leafSize,
			InteriorsRawRepr interiors, LeavesRawRepr leaves, const AABB& boundary,
			RangeQueryResponsesRawRepr resps);
	};

	struct UpdateKernel {
		static void findLeafBin(int qIdx, int qSize, const vec3f* qPts, int leafSize,
			const InteriorsRawRepr interiors, const LeavesRawRepr leaves, const AABB& boundary,
			OUTPUT(int*) binIdx);
	};

	struct VerifyKernel {

	};

}