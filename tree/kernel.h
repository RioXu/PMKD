#pragma once
#include <vector>
#include <atomic>

#ifdef ENABLE_MERKLE
#include <auth/verification_node.h>
#endif
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
		
		static void calcBuildMetrics(int idx, int interiorSize, const AABB& gBoundary, INPUT(MortonType*) morton,
			OUTPUT(uint8_t*) metrics, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal);

		static void buildInteriors(int idx, int leafSize, const LeavesRawRepr leaves,
			InteriorsRawRepr interiors, BuildAid aid);

		// optimized version of buildInteriors by removing branches
		static void buildInteriors_opt(int idx, int leafSize, const LeavesRawRepr leaves, INPUT(int*) metrics,
			OUTPUT(int*) range[2], OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal,
			OUTPUT(int*) parent, BuildAid aid);

		static void calcInteriorNewIdx(int idx, int size, const LeavesRawRepr leaves, const InteriorsRawRepr interiors,
			INPUT(int*) segLen, INPUT(int*) leftLeafCount, OUTPUT(int*) mapidx);

		// in place
		static void reorderInteriors(int idx, int interiorSize, INPUT(int*) mapidx, const InteriorsRawRepr interiors,
			OUTPUT(int*) rangeL, OUTPUT(int*) rangeR, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal, OUTPUT(int*) parent);

		static void remapLeafParents(int idx, int leafSize, INPUT(int*) mapidx, LeavesRawRepr leaves);

#ifdef ENABLE_MERKLE
		static void calcLeafHash(int idx, int size, INPUT(vec3f*) pts, INPUT(int*) removeFlag, OUTPUT(hash_t*) leafHash);

		static void calcLeafHash(int idx, int size, INPUT(vec3f*) pts, OUTPUT(hash_t*) leafHash);

		static void calcInteriorHash(int idx, int leafSize, const LeavesRawRepr leaves,
			InteriorsRawRepr interiors, OUTPUT(AtomicCount*) visitCount);
#endif
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

		static void reorderInteriors(int idx, int batchInteriorSize, INPUT(int*) mapidx, const InteriorsRawRepr interiors,
			OUTPUT(int*) rangeL, OUTPUT(int*) rangeR, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal, OUTPUT(int*) parent);

		static void remapLeafParents(int idx, int batchLeafSize, INPUT(int*) mapidx, LeavesRawRepr leaves);

		static void setSubtreeRootParentSplit(int idx, int numSubTree,
			INPUT(int*) interiorCount, INPUT(int*) derivedFrom, const NodeMgrDevice nodeMgr, const AABB& gBoundary,
			OUTPUT(int*) parentSplitDim, OUTPUT(mfloat*) parentSplitVal);

#ifdef ENABLE_MERKLE
		static void calcInteriorHash_Batch(int idx, int batchLeafSize,
			const LeavesRawRepr leaves, InteriorsRawRepr interiors);

		static void calcInteriorHash_Upper(int idx, int numSubTree, INPUT(int*) interiorCount, INPUT(int*) binIdx,
			const InteriorsRawRepr interiors, NodeMgrDevice nodeMgr);

		static void calcInteriorHash_Full(int idx, int batchLeafSize, const LeavesRawRepr leaves,
			InteriorsRawRepr interiors, NodeMgrDevice nodeMgr);
#endif
	};


	struct SearchKernel {
		static void searchPoints(int qIdx, int qSize, const Query* qPts, const vec3f* pts, int leafSize,
			const InteriorsRawRepr interiors, const LeavesRawRepr leaves, const AABB& boundary, uint8_t* exist);

		static void searchPoints(int qIdx, int qSize, const Query* qPts, const NodeMgrDevice nodeMgr, int totalLeafSize,
			const AABB& boundary, uint8_t* exist);

		static void searchRanges(int qIdx, int qSize, const RangeQuery* qRanges, const vec3f* pts, int leafSize,
			const InteriorsRawRepr interiors, const LeavesRawRepr leaves, const AABB& boundary,
			RangeQueryResponsesRawRepr resps);

		static void searchRanges(int qIdx, int qSize, const RangeQuery* qRanges, const NodeMgrDevice nodeMgr, int totalLeafSize,
			const AABB& boundary, RangeQueryResponsesRawRepr resps);

#ifdef ENABLE_MERKLE
		static void searchRangesVerifiable_step1(int qIdx, int qSize, const RangeQuery* qRanges, const NodeMgrDevice nodeMgr, int totalLeafSize,
			OUTPUT(int*) fCnt, OUTPUT(int*) mCnt, OUTPUT(int*) hCnt);

		static void searchRangesVerifiable_step2(int qIdx, int qSize, const RangeQuery* qRanges, const NodeMgrDevice nodeMgr, int totalLeafSize,
			INPUT(int*) fOffset, INPUT(int*) mOffset, INPUT(int*) hOffset,
			FNodesRawRepr fNodes, MNodesRawRepr mNodes, HNodesRawRepr hNodes);
#endif
	};

	struct UpdateKernel {
		// for insertion
		static void findLeafBin(int qIdx, int qSize, const vec3f* qPts, int leafSize,
			const InteriorsRawRepr interiors, const LeavesRawRepr leaves,
			OUTPUT(int*) binIdx);

		static void findLeafBin(int qIdx, int qSize, const vec3f* qPts, int leafSize,
			const InteriorsRawRepr interiors, const LeavesRawRepr leaves,
			OUTPUT(int*) binIdx, std::atomic<int>* maxBin);

		static void findLeafBin(int qIdx, int qSize, const vec3f* qPts, int totalLeafSize,
			const NodeMgrDevice nodeMgr, OUTPUT(int*) binIdx);

		static void revertRemoval(int qIdx, int qSize, INPUT(int*) binIdx, NodeMgrDevice nodeMgr);
#ifdef ENABLE_MERKLE
		// depricated
		static void updateMerkleHash(int mIdx, int mSize, INPUT(int*) mixOpBinIdx, NodeMgrDevice nodeMgr);
#endif
		// for removal
		static void removePoints_step1(int rIdx, int rSize, const vec3f* rPts, const vec3f* pts, int leafSize,
			InteriorsRawRepr interiors, LeavesRawRepr leaves, OUTPUT(int*) binIdx);

		static void removePoints_step1(int rIdx, int rSize, const vec3f* rPts, const NodeMgrDevice nodeMgr,
			int totalLeafSize, OUTPUT(int*) binIdx);

#ifdef ENABLE_MERKLE
		static void calcSelectedLeafHash(int rIdx, int rSize, INPUT(int*) binIdx,
			NodeMgrDevice nodeMgr);
#endif

		static void removePoints_step2(int rIdx, int rSize, int leafSize, INPUT(int*) binIdx, const LeavesRawRepr leaves,
			InteriorsRawRepr interiors);
		
		static void removePoints_step2(int rIdx, int rSize, INPUT(int*) binIdx,	NodeMgrDevice nodeMgr);

		// removal v2
		static void removePoints_v2_step1(int rIdx, int rSize, const vec3f* rPts, const vec3f* pts, int leafSize,
			InteriorsRawRepr interiors, LeavesRawRepr leaves);
	};

	struct VerifyKernel {

	};

}