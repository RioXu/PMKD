#include <tree/device_helper.h>
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
    }

    void DynamicBuildKernel::buildInteriors(int idx, int batchLeafSize, INPUT(int*) localRangeL, const LeavesRawRepr leaves,
        InteriorsRawRepr interiors, BuildAid aid) {

        if (idx >= batchLeafSize) return;
        int lBound = localRangeL[idx];
        int rBound = leaves.treeLocalRangeR[idx];

        int parent;
        bool isLeftMost = idx == lBound;
        bool isRightMost = idx == rBound - 1;
        bool isRC = !isLeftMost && (isRightMost || aid.metrics[idx - 1] <= aid.metrics[idx]);
        if (!isRC) {
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
        leaves.parent[idx] = encodeParentCode(parent, isRC);

        // choose parent in bottom-up fashion. O(n)
        int current, left, right;
        while (setVisitCountBottomUp(aid.visitCount[parent], isRC)) {
            current = parent;

            left = interiors.rangeL[current];
            right = interiors.rangeR[current];

            isLeftMost = left == lBound;
            isRightMost = right == rBound - 1;
            if (isLeftMost && isRightMost) {
                //root
                interiors.parent[current] = -1;
                break;
            }

            isRC = !isLeftMost && (isRightMost || aid.metrics[left - 1] <= aid.metrics[right]);
            if (!isRC) {
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
            interiors.parent[current] = encodeParentCode(parent, isRC);
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

    void DynamicBuildKernel::reorderInteriors(int idx, int batchInteriorSize, INPUT(int*) mapidx, const InteriorsRawRepr interiors,
        OUTPUT(int*) rangeL, OUTPUT(int*) rangeR, OUTPUT(int*) splitDim, OUTPUT(mfloat*) splitVal, OUTPUT(int*) parent) {

        if (idx >= batchInteriorSize) return;
        int mapped_idx = mapidx[idx];
        if (mapped_idx == -1) return;

        rangeL[mapped_idx] = interiors.rangeL[idx];
        rangeR[mapped_idx] = interiors.rangeR[idx];

        splitDim[mapped_idx] = interiors.splitDim[idx];
        splitVal[mapped_idx] = interiors.splitVal[idx];

        int parentIdx;
        bool isRC;
        decodeParentCode(interiors.parent[idx], parentIdx, isRC);
        parent[mapped_idx] = parentIdx >= 0 ? encodeParentCode(mapidx[parentIdx], isRC) : -1;
    }

    void DynamicBuildKernel::remapLeafParents(int idx, int batchLeafSize, INPUT(int*) mapidx, LeavesRawRepr leaves) {

        if (idx >= batchLeafSize) return;

        int parentIdx;
        bool isRC;
        decodeParentCode(leaves.parent[idx], parentIdx, isRC);
        int mapped_idx = mapidx[parentIdx];
        //assert(mapped_idx != -1);

        leaves.parent[idx] = encodeParentCode(mapped_idx, isRC);
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

#ifdef ENABLE_MERKLE
    void DynamicBuildKernel::calcInteriorHash_Batch(int idx, int batchLeafSize, 
        const LeavesRawRepr leaves, InteriorsRawRepr interiors) {
        if (idx >= batchLeafSize) return;

        int rBound = leaves.treeLocalRangeR[idx];

        int parent;
        bool isRC;
        decodeParentCode(leaves.parent[idx], parent, isRC);

        int current = idx;
        const hash_t* childHash = leaves.hash + current;

        // choose parent in bottom-up fashion. O(n)
        while (setVisitStateBottomUpAsVisitCount(interiors.visitState[parent], isRC)) {
            current = parent;

            int left = interiors.rangeL[current];
            int right = interiors.rangeR[current];

            const hash_t* otherChildHash;
            getOtherChildHash(leaves, interiors, left, current, rBound, isRC, otherChildHash);

            computeDigest(interiors.hash + current, childHash, otherChildHash,
                interiors.splitDim[current], interiors.splitVal[current], interiors.removeState[current].load(std::memory_order_relaxed));

            int parentCode = interiors.parent[current];
            if (parentCode < 0) {
                //root
                break;
            }

            childHash = interiors.hash + current;

            decodeParentCode(parentCode, parent, isRC);
        }
    }

    void DynamicBuildKernel::calcInteriorHash_Upper(int idx, int numSubTree, INPUT(int*) interiorCount, INPUT(int*) binIdx,
        const InteriorsRawRepr interiorsInsert, NodeMgrDevice nodeMgr) {
        if (idx >= numSubTree) return;

        int interiorIdx = interiorCount[idx];
        int mainTreeLeafSize = nodeMgr.sizesAcc[0];

        const hash_t* childHash = &interiorsInsert.hash[interiorIdx];

        int globalLeafIdx = binIdx[idx];
        int iBatch, localLeafIdx;
        transformLeafIdx(globalLeafIdx, nodeMgr.sizesAcc, nodeMgr.numBatches, iBatch, localLeafIdx);

        while (true) {
            const auto& leaves = nodeMgr.leavesBatch[iBatch];
            auto& interiors = nodeMgr.interiorsBatch[iBatch];

            int rBound = iBatch == 0 ? mainTreeLeafSize : leaves.treeLocalRangeR[localLeafIdx];

            int parent;
            bool isRC;
            decodeParentCode(leaves.parent[localLeafIdx], parent, isRC);

            int current = -1;
            // choose parent in bottom-up fashion. O(n)
            while (setVisitStateBottomUp(interiors.visitStateTopDown, parent, interiors.visitState[parent], isRC))
            {
                clearVisitStateTopDown(interiors.visitStateTopDown, parent);
                current = parent;

                int LR[2] = { interiors.rangeL[current] ,interiors.rangeR[current] };  // left, right
                const auto& left = LR[0];
                const auto& right = LR[1];

                const hash_t* otherChildHash;
                getOtherChildHash(leaves, interiors, left, current, rBound, isRC, otherChildHash);

                computeDigest(interiors.hash + current, childHash, otherChildHash,
                    interiors.splitDim[current], interiors.splitVal[current], interiors.removeState[current].load(std::memory_order_relaxed));

                childHash = interiors.hash + current;

                int parentCode = interiors.parent[current];
                decodeParentCode(parentCode, parent, isRC);

                if (parentCode < 0) {
                    break;
                }
            }
            if (current != parent || iBatch == 0) break;  // does not reach sub root, or main root visited

            globalLeafIdx = leaves.derivedFrom[localLeafIdx];
            transformLeafIdx(globalLeafIdx, nodeMgr.sizesAcc, nodeMgr.numBatches, iBatch, localLeafIdx);
        }
    }

    void DynamicBuildKernel::calcInteriorHash_Full(int idx, int batchLeafSize, const LeavesRawRepr newLeaves,
        InteriorsRawRepr newInteriors, NodeMgrDevice nodeMgr) {
        if (idx >= batchLeafSize) return;

        int mainTreeLeafSize = nodeMgr.sizesAcc[0];

        const LeavesRawRepr* leaves = &newLeaves;
        InteriorsRawRepr* interiors = &newInteriors;

        int rBound = newLeaves.treeLocalRangeR[idx];

        int parent;
        bool isRC;
        decodeParentCode(leaves->parent[idx], parent, isRC);

        const hash_t* childHash = newLeaves.hash + idx;

        bool isUpperLevel = false;
        bool canMoveOn = setVisitStateBottomUpAsVisitCount(newInteriors.visitState[parent], isRC);
        int globalLeafIdx, localLeafIdx = idx;
        int iBatch = -1;

        int current = -1;
        while (true) {
            // choose parent in bottom-up fashion. O(n)
            while (canMoveOn)
            {
                if (isUpperLevel) clearVisitStateTopDown(interiors->visitStateTopDown, parent);
                current = parent;

                int left = interiors->rangeL[current];
                int right = interiors->rangeR[current];

                const hash_t* otherChildHash;
                getOtherChildHash(*leaves, *interiors, left, current, rBound, isRC, otherChildHash);

                computeDigest(interiors->hash + current, childHash, otherChildHash,
                    interiors->splitDim[current], interiors->splitVal[current], interiors->removeState[current].load(std::memory_order_relaxed));

                childHash = interiors->hash + current;

                int parentCode = interiors->parent[current];

                if (parentCode < 0) {
                    break;
                }

                decodeParentCode(parentCode, parent, isRC);

                if (!isUpperLevel) canMoveOn = setVisitStateBottomUpAsVisitCount(interiors->visitState[parent], isRC);
                else canMoveOn = setVisitStateBottomUp(interiors->visitStateTopDown, parent, interiors->visitState[parent], isRC);
            }
            if (current != parent || iBatch == 0) break;  // does not reach sub root, or main root visited

            isUpperLevel = true;

            globalLeafIdx = leaves->derivedFrom[localLeafIdx];
            transformLeafIdx(globalLeafIdx, nodeMgr.sizesAcc, nodeMgr.numBatches, iBatch, localLeafIdx);

            leaves = nodeMgr.leavesBatch + iBatch;
            interiors = nodeMgr.interiorsBatch + iBatch;

            rBound = leaves->treeLocalRangeR[localLeafIdx];

            decodeParentCode(leaves->parent[localLeafIdx], parent, isRC);
            current = -1;

            canMoveOn = setVisitStateBottomUp(interiors->visitStateTopDown, parent, interiors->visitState[parent], isRC);
        }
    }
#endif
}