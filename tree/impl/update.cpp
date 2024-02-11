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
                    uint32_t globalSubstitute = leaves.replacedBy[localLeafIdx];
                    if (globalSubstitute == 0) { // this leaf is valid (i.e. not replaced)
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
}