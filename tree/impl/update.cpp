#include <common/util/atomic_ext.h>
#include <tree/helper.h>
#include <tree/kernel.h>

namespace pmkd {
    
    void UpdateKernel::findLeafBin(int qIdx, int qSize, const vec3f* qPts, int leafSize,
        const InteriorsRawRepr interiors, const LeavesRawRepr leaves,
        OUTPUT(int*) binIdx) {

        if (qIdx >= qSize) return;

        const vec3f& pt = qPts[qIdx];

        // do sprouting
        int L = 0, R = 0;
        int interiorIdx = 0;

        //mfloat ptVal_256[8];
        //int onRight_256[8];
        bool onRight;

        for (int bin = 0; bin < leafSize; bin++) {
            L = leaves.segOffset[bin];
            R = bin == leafSize - 1 ? L : leaves.segOffset[bin + 1];
            onRight = false;

            // if (L + 8 <= R) {
            //     memset(onRight_256, 0, 8 * sizeof(int));
            //     for (interiorIdx = L; interiorIdx + 8 <= R; interiorIdx += 8) {
            //         for (int j = 0;j < 8;j++) ptVal_256[j] = pt[interiors.splitDim[interiorIdx + j]];
            //         compare_vectors(ptVal_256, interiors.splitVal + interiorIdx, onRight_256);

            //         for (int j = 0;j < 8;j++) {
            //             if (onRight_256[j] > 0) {
            //                 onRight = true;
            //                 interiorIdx += j;
            //                 break;
            //             }
            //         }
            //         if (onRight) break;
            //     }
            // }
            //else {
                for (interiorIdx = L; interiorIdx < R; interiorIdx++) {
                    int splitDim = interiors.splitDim[interiorIdx];
                    mfloat splitVal = interiors.splitVal[interiorIdx];
                    onRight = pt[splitDim] >= splitVal;
                    if (onRight) {
                        break;
                    }
                }
            //}

            if (!onRight) { // bin found
                binIdx[qIdx] = bin;
                break;
            }
            bin = interiorIdx < R - 1 ? interiors.rangeR[interiorIdx + 1] : bin;
        }
    }

    void UpdateKernel::findLeafBin(int qIdx, int qSize, const vec3f* qPts, int leafSize,
        const InteriorsRawRepr interiors, const LeavesRawRepr leaves,
        OUTPUT(int*) binIdx, std::atomic<int>* maxBin) {

        if (qIdx >= qSize) return;

        const vec3f& pt = qPts[qIdx];

        // do sprouting
        int L = 0, R = 0;
        int interiorIdx = 0;
        bool onRight;

        for (int bin = 0; bin < leafSize; bin++) {
            L = leaves.segOffset[bin];
            R = bin == leafSize - 1 ? L : leaves.segOffset[bin + 1];
            onRight = false;

            for (interiorIdx = L; interiorIdx < R; interiorIdx++) {
                int splitDim = interiors.splitDim[interiorIdx];
                mfloat splitVal = interiors.splitVal[interiorIdx];
                onRight = pt[splitDim] >= splitVal;
                if (onRight) {
                    break;
                }
            }

            if (!onRight) { // bin found
                binIdx[qIdx] = bin;
                atomic_fetch_max_explicit(maxBin, bin, std::memory_order_relaxed);  // note: not sure if relax order is ok
                break;
            }
            bin = interiorIdx < R - 1 ? interiors.rangeR[interiorIdx + 1] : bin;
        }
    }

    void UpdateKernel::findLeafBin(int qIdx, int qSize, const vec3f* qPts, int totalLeafSize,
        const NodeMgrDevice nodeMgr, OUTPUT(int*) binIdx) {
        if (qIdx >= qSize) return;
        const vec3f& pt = qPts[qIdx];

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

            int rBound = iBatch == 0 ? mainTreeLeafSize : leaves.treeLocalRangeR[localLeafIdx];

            while (localLeafIdx < rBound) {
                int oldLocalLeafIdx = localLeafIdx;

                L = leaves.segOffset[localLeafIdx];
                R = localLeafIdx == rBound - 1 ? L : leaves.segOffset[localLeafIdx + 1];
                onRight = false;
                for (interiorIdx = L; interiorIdx < R; interiorIdx++) {
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
                        binIdx[qIdx] = globalLeafIdx;
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

    inline void setVisitFlag(AtomicCount& alterState, bool fromRC) {
        alterState.cnt |= (1 << fromRC);
    }

    // inline bool unsetVisitFlag(AtomicCount& alterState, bool fromRC) {
    //     int bit = 1 << fromRC;
    //     int oldState = alterState.cnt.fetch_xor(bit) & 0b11;
    //     //assert(((oldState>>fromRC)&1) == 1);
    //     //return (alterState.cnt.fetch_xor(bit) & 0b11) == bit;
    //     return oldState == bit;
    // }

    // return: is unvisited after modification
    inline bool unsetVisitFlag(AtomicCount& alterState, bool fromRC) {
        int bit = 1 << fromRC;
        int oldState = alterState.cnt.fetch_and(~bit) & 0b11;
        return oldState == bit;
    }

    // return: is removed after modification
    // inline bool setRemoveFlag(AtomicCount& alterState, bool fromRC) {
    //     int bit = 0b100 << fromRC;
    //     return (alterState.cnt.fetch_or(bit) & 0b1100) == 0b1100;
    // }
    inline void setRemoveFlag(AtomicCount& alterState, bool fromRC) {
        alterState.cnt |= (0b100 << fromRC);
    }

    inline bool isInteriorRemoved(const AtomicCount& alterState) {
        int state = alterState.cnt.load(std::memory_order_relaxed);
        return (state & 0b1100) == 0b1100;
    }

    void UpdateKernel::removePoints_step1(int rIdx, int rSize, const vec3f* rPts, const vec3f* pts, int leafSize,
        InteriorsRawRepr interiors, LeavesRawRepr leaves, OUTPUT(int*) binIdx) {
        
        if (rIdx >= rSize) return;
        const vec3f& pt = rPts[rIdx];

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

                setVisitFlag(interiors.alterState[interiorIdx], onRight);
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
                leaves.replacedBy[begin] = -1;  // mark as removed
                binIdx[rIdx] = begin;
                break;
            }
        }
    }

    void UpdateKernel::removePoints_step1(int rIdx, int rSize, const vec3f* rPts, const NodeMgrDevice nodeMgr, int totalLeafSize, OUTPUT(int*) binIdx) {

        if (rIdx >= rSize) return;
        const vec3f& pt = rPts[rIdx];

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
                    int splitDim = interiors.splitDim[interiorIdx];
                    mfloat splitVal = interiors.splitVal[interiorIdx];
                    onRight = pt[splitDim] >= splitVal;

                    setVisitFlag(interiors.alterState[interiorIdx], onRight);
                    if (onRight) {
                        // goto right child
                        localLeafIdx = interiorIdx < R - 1 ? interiors.rangeR[interiorIdx + 1] : localLeafIdx;
                        break;
                    }
                }
                if (!onRight) {
                    int globalSubstitute = leaves.replacedBy[localLeafIdx];
                    if (globalSubstitute <= 0) { // this leaf is valid or removed (i.e. not replaced)
                        leaves.replacedBy[localLeafIdx] = -1;  // mark as removed
                        binIdx[rIdx] = globalLeafIdx;
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

    void UpdateKernel::removePoints_step2(int rIdx, int rSize, int leafSize, INPUT(int*) binIdx, const LeavesRawRepr leaves,
        InteriorsRawRepr interiors) {
        if (rIdx >= rSize) return;
        int idx = binIdx[rIdx];

        bool isRC = idx != 0 && (idx == leafSize - 1 || interiors.metrics[idx - 1] <= interiors.metrics[idx]);
        int parent = interiors.mapidx[idx - isRC];
        setRemoveFlag(interiors.alterState[parent], isRC);

        // choose parent in bottom-up fashion. O(n)
        int current, left, right;
        while (unsetVisitFlag(interiors.alterState[parent], isRC)) {
            current = parent;

            if (current == 0) break; // root

            int LR[2] = { interiors.rangeL[current] ,interiors.rangeR[current] };  // left, right
            const auto& left = LR[0];
            const auto& right = LR[1];

            isRC = left != 0 && (right == leafSize - 1 || interiors.metrics[left - 1] <= interiors.metrics[right]);
            parent = interiors.mapidx[LR[1 - isRC] - isRC];

            if (isInteriorRemoved(interiors.alterState[current]))
                setRemoveFlag(interiors.alterState[parent], isRC);
        }
    }

    void UpdateKernel::removePoints_step2(int rIdx, int rSize, int totalLeafSize, INPUT(int*) binIdx, NodeMgrDevice nodeMgr) {
        
        if (rIdx >= rSize) return;
        int globalLeafIdx = binIdx[rIdx];

        int mainTreeLeafSize = nodeMgr.sizesAcc[0];

        bool isSubRootRemoved = true;

        while (true) {
            int iBatch, localLeafIdx;
            transformLeafIdx(globalLeafIdx, nodeMgr.sizesAcc, nodeMgr.numBatches, iBatch, localLeafIdx);
            const auto& leaves = nodeMgr.leavesBatch[iBatch];
            const auto& interiors = nodeMgr.interiorsBatch[iBatch];

            int rBound = iBatch == 0 ? mainTreeLeafSize : nodeMgr.leavesBatch[iBatch].treeLocalRangeR[localLeafIdx];
            bool isLeftMost = localLeafIdx == 0 || (interiors.mapidx[localLeafIdx - 1] == -1);
            bool isRightMost = localLeafIdx == rBound - 1;

            bool isRC = !isLeftMost && (isRightMost || interiors.metrics[localLeafIdx - 1] <= interiors.metrics[localLeafIdx]);
            int parent = interiors.mapidx[localLeafIdx - isRC];
            if (isSubRootRemoved)
                setRemoveFlag(interiors.alterState[parent], isRC);
            
            // choose parent in bottom-up fashion. O(n)
            int current, left, right;
            while (unsetVisitFlag(interiors.alterState[parent], isRC)) {
                current = parent;

                int LR[2] = { interiors.rangeL[current] ,interiors.rangeR[current] };  // left, right
                const auto& left = LR[0];
                const auto& right = LR[1];

                isLeftMost = left == 0 || (interiors.mapidx[left - 1] == -1);
                isRightMost = right == rBound - 1;

                if (isLeftMost && isRightMost) {
                    break;
                }

                isRC = !isLeftMost && (isRightMost || interiors.metrics[left - 1] <= interiors.metrics[right]);
                parent = interiors.mapidx[LR[1 - isRC] - isRC];

                if (isInteriorRemoved(interiors.alterState[current]))
                    setRemoveFlag(interiors.alterState[parent], isRC);
            }
            if (current != parent || iBatch == 0) break;  // does not reach sub root, or main root visited

            isSubRootRemoved = isInteriorRemoved(interiors.alterState[current]);
            globalLeafIdx = leaves.derivedFrom[localLeafIdx];
        }
    }
}