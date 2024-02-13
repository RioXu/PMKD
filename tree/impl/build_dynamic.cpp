#include <tree/kernel.h>

namespace pmkd {
    void DynamicBuildKernel::calcBuildMetrics(int idx, int interiorRealSize, const AABB& gBoundary,
        INPUT(MortonType*) morton, INPUT(int*) interiorToLeafIdx,
        OUTPUT(uint8_t*) metrics, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal) {
        
        if (idx >= interiorRealSize) return;
        int leafIdx = interiorToLeafIdx[idx];

        // reflects the highest differing bit between the keys covered by interior node idx
        uint8_t metric = MortonType::calcMetric(morton[leafIdx], morton[leafIdx + 1]);
        metrics[leafIdx] = metric;
        MortonType::calcSplit(metric, morton[leafIdx + 1], gBoundary.ptMin, gBoundary.ptMax,
            splitDim + leafIdx, splitVal + leafIdx);
        int flag=1;
    }

    void DynamicBuildKernel::buildInteriors(int idx, int batchLeafSize, INPUT(int*) localRangeL, const LeavesRawRepr leaves,
        InteriorsRawRepr interiors, BuildAid aid) {

        if (idx >= batchLeafSize) return;

        int parent;
        bool isLeftMost = idx == localRangeL[idx];
        bool isRightMost = idx == leaves.treeLocalRangeR[idx] - 1;
        if (isLeftMost || (!isRightMost && aid.metrics[idx - 1] > aid.metrics[idx])) {
            // is left child of Interior idx
            parent = idx;
            interiors.rangeL[parent] = idx;
            aid.segLen[idx] = 1;
            aid.leftLeafCount[idx] = 1;
        }
        else {
            // is right child of Interior idx-1
            parent = idx - 1;
            interiors.rangeR[parent] = idx;
        }

        // choose parent in bottom-up fashion. O(n)
        int current, left, right;
        while (aid.visitCount[parent].cnt++ == 1) {
            current = parent;

            left = interiors.rangeL[current];
            right = interiors.rangeR[current];

            isLeftMost = left == localRangeL[left];
            isRightMost = right == leaves.treeLocalRangeR[right] - 1;
            if (isLeftMost && isRightMost) {
                //root
                interiors.parentSplitDim[current] = -1;
                break;
            }

            if (isLeftMost || (!isRightMost && aid.metrics[left - 1] > aid.metrics[right])) {
                // is left child of Interior right
                parent = right;
                interiors.rangeL[parent] = left;

                aid.segLen[left]++;  // add LCL value of the leftmost leaf
                aid.leftLeafCount[parent] = aid.segLen[left]; // count the order of Interior parent
            }
            else {
                // is right child of Interior left-1
                parent = left - 1;
                interiors.rangeR[parent] = right;
            }
            interiors.parentSplitDim[current] = interiors.splitDim[parent];
            interiors.parentSplitVal[current] = interiors.splitVal[parent];
        }
    }

    void DynamicBuildKernel::interiorMapIdxInit(int idx, int numSubTree, int batchLeafSize, INPUT(int*) interiorCount,
        OUTPUT(int*) mapidx) {

        if (idx >= numSubTree - 1) return;
        //int j = idx < numSubTree - 1 ? interiorCount[idx + 1] + idx : batchLeafSize - 1;
        int j = interiorCount[idx + 1] + idx;
        mapidx[j] = -1;
    }

    void DynamicBuildKernel::calcInteriorNewIdx(int idx, int interiorRealSize, INPUT(int*) interiorToLeafIdx,
        const LeavesRawRepr leaves, const InteriorsRawRepr interiors,
        INPUT(int*) segLen, INPUT(int*) leftLeafCount, OUTPUT(int*) mapidx) {

        if (idx >= interiorRealSize) return;
        int j = interiorToLeafIdx[idx];

        int binIdx = interiors.rangeL[j];
        int idx_new = leaves.segOffset[binIdx] + segLen[binIdx] - leftLeafCount[j];
        mapidx[j] = idx_new;
    }

    void DynamicBuildKernel::reorderInteriors_step1(int idx, int batchInteriorSize, const InteriorsRawRepr interiors, INPUT(int*) mapidx,
        OUTPUT(int*) rangeL, OUTPUT(int*) rangeR, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal) {

        if (idx >= batchInteriorSize) return;
        int mapped_idx = mapidx[idx];
        if (mapped_idx == -1) return;

        rangeL[mapped_idx] = interiors.rangeL[idx];
        rangeR[mapped_idx] = interiors.rangeR[idx];

        splitDim[mapped_idx] = interiors.splitDim[idx];
        splitVal[mapped_idx] = interiors.splitVal[idx];
    }

    void DynamicBuildKernel::reorderInteriors_step2(int idx, int batchInteriorSize, const InteriorsRawRepr interiors, INPUT(int*) mapidx,
        OUTPUT(int*) parentSplitDim, OUTPUT(mfloat*) parentSplitVal) {

        if (idx >= batchInteriorSize) return;
        int mapped_idx = mapidx[idx];
        if (mapped_idx == -1) return;

        parentSplitDim[mapped_idx] = interiors.parentSplitDim[idx];
        parentSplitVal[mapped_idx] = interiors.parentSplitVal[idx];
    }

    void DynamicBuildKernel::setSubtreeRootParentSplit(int idx, int numSubTree,
        INPUT(int*) interiorCount, INPUT(int*) derivedFrom, const NodeMgrDevice nodeMgr, const AABB& gBoundary,
        OUTPUT(int*) parentSplitDim, OUTPUT(mfloat*) parentSplitVal) {

        if (idx >= numSubTree) return;
        int subtreeRoot = interiorCount[idx];
        // int firstLeafOfSubtree = subtreeRoot + idx;
        // int binIdx = derivedFrom[firstLeafOfSubtree];

        // int iBatch, offset;
        // transformLeafIdx(binIdx, nodeMgr.sizesAcc, nodeMgr.numBatches, iBatch, offset);
        // const auto& leaves = nodeMgr.leavesBatch[iBatch];
        // const auto& interiors = nodeMgr.interiorsBatch[iBatch];

        // int splitDim;
        // mfloat splitVal;
        // // judge if binIdx is left leaf or right leaf
        // int L = leaves.segOffset[offset];
        // int R = binIdx < nodeMgr.sizesAcc[iBatch] - 1 ? leaves.segOffset[offset + 1] : L;

        // const auto& pt = nodeMgr.ptsBatch[iBatch][offset];

        // if (L < R) {
        //     // is left leaf
        //     splitDim = interiors.splitDim[R - 1];
        //     splitVal = interiors.splitVal[R - 1];
        // }
        // else {
        //     // is right leaf
        //     int metric = MortonType::calcMetric(leaves.morton[offset-1], leaves.morton[offset]);

        //     MortonType::calcSplit(metric, leaves.morton[offset], gBoundary.ptMin, gBoundary.ptMax,
        //         &splitDim, &splitVal);
        // }
        parentSplitDim[subtreeRoot] = -1;
        // parentSplitVal[subtreeRoot] = splitVal;
    }
}