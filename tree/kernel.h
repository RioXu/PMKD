#pragma once
#include <vector>
#include <atomic>
#include "node.h"
#include "query_response.h"

namespace pmkd {
#define INPUT(ptr_t) const ptr_t __restrict_arr
#define OUTPUT(ptr_t) ptr_t __restrict_arr

	struct BuildKernel {
		static void reduceBoundary(int idx, int size, INPUT(vec3f*) pts, OUTPUT(AABB*) boundary);

		static MortonType calcMortonCode(const vec3f& pt, const AABB& boundary) {
			vec3f offset = (pt - boundary.ptMin);
			offset.x /= (boundary.ptMax.x - boundary.ptMin.x);
			offset.y /= (boundary.ptMax.y - boundary.ptMin.y);
			offset.z /= (boundary.ptMax.z - boundary.ptMin.z);
			return MortonType::calculate(offset.x, offset.y, offset.z);
		}

		static void calcMortonCodes(int idx, int size, INPUT(vec3f*) pts, INPUT(AABB*) gboundary,
			OUTPUT(MortonType*) morton);

		//static void reorderLeaves(int idx, int size, LeavesRawRepr&& leaves, LeavesRawRepr&& leaves_sorted, int* mapidx);

		static void calcBuildMetrics(int idx, int interiorSize, const AABB& gBoundary, INPUT(MortonType*) morton,
			OUTPUT(uint8_t*) metrics, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal);

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

	
	struct DynamicBuildKernel {
		static void calcBuildMetrics(int idx, int interiorRealSize, const AABB& gBoundary,
			INPUT(MortonType*) morton, INPUT(int*) interiorToLeafIdx,
			OUTPUT(uint8_t*) metrics, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal);

		static void buildInteriors(int idx, int batchLeafSize, INPUT(int*) localRangeL, const LeavesRawRepr leaves,
			InteriorsRawRepr interiors, BuildAid aid);

		static void interiorMapIdxInit(int idx, int numSubTree, int batchLeafSize, INPUT(int*) interiorCount,
			OUTPUT(int*) mapidx);

		static void calcInteriorNewIdx(int idx, int interiorRealSize, INPUT(int*) interiorToLeafIdx,
			const LeavesRawRepr leaves, const InteriorsRawRepr interiors,
			INPUT(int*) segLen, INPUT(int*) leftLeafCount, OUTPUT(int*) mapidx);

		static void reorderInteriors_step1(int idx, int batchInteriorSize, const InteriorsRawRepr interiors, INPUT(int*) mapidx,
			OUTPUT(int*) rangeL, OUTPUT(int*) rangeR, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal);

		static void reorderInteriors_step2(int idx, int batchInteriorSize, const InteriorsRawRepr interiors, INPUT(int*) mapidx,
			OUTPUT(int*) parentSplitDim, OUTPUT(mfloat*) parentSplitVal);

		static void setSubtreeRootParentSplit(int idx, int numSubTree,
			INPUT(int*) interiorCount, INPUT(int*) derivedFrom, const NodeMgrDevice nodeMgr, const AABB& gBoundary,
			OUTPUT(int*) parentSplitDim, OUTPUT(mfloat*) parentSplitVal);
	};


	struct SearchKernel {
		static void searchPoints(int qIdx, int qSize, const Query* qPts, const vec3f* pts, int leafSize,
			InteriorsRawRepr interiors, LeavesRawRepr leaves, const AABB& boundary, bool* exist);

		static void searchPoints(int qIdx, int qSize, const Query* qPts, const NodeMgrDevice nodeMgr, int totalLeafSize,
			const AABB& boundary, bool* exist);

		static void searchRanges(int qIdx, int qSize, const RangeQuery* qRanges, const vec3f* pts, int leafSize,
			InteriorsRawRepr interiors, LeavesRawRepr leaves, const AABB& boundary,
			RangeQueryResponsesRawRepr resps);

		static void searchRanges(int qIdx, int qSize, const RangeQuery* qRanges, const NodeMgrDevice nodeMgr, int totalLeafSize,
			const AABB& boundary, RangeQueryResponsesRawRepr resps);
	};

	struct UpdateKernel {
		static void findLeafBin(int qIdx, int qSize, const vec3f* qPts, int leafSize,
			const InteriorsRawRepr interiors, const LeavesRawRepr leaves,
			OUTPUT(int*) binIdx);

		static void findLeafBin(int qIdx, int qSize, const vec3f* qPts, int leafSize,
			const InteriorsRawRepr interiors, const LeavesRawRepr leaves,
			OUTPUT(int*) binIdx, std::atomic<int>* maxBin);

		static void findLeafBin(int qIdx, int qSize, const vec3f* qPts, int totalLeafSize,
			const NodeMgrDevice nodeMgr, OUTPUT(int*) binIdx);

		static void removePoints(int rIdx, int rSize, const vec3f* rPts, const NodeMgrDevice nodeMgr, int totalLeafSize,
			const AABB& boundary);
	};

	struct VerifyKernel {

	};

}