#pragma once
#include <auth/sha.h>
#include <node.h>

namespace pmkd {
    // inline int toKey(int idx, uint8_t isInterior) {
    //     return (idx << 1) + isInterior;
    // }

    // inline int toFKey(int idx) {
    //     return idx << 1;
    // }

    // inline int toMKey(int idx) {
    //     return (idx << 1) + 1;
    // }

    // inline int toHKey(int idx) {
    //     return (idx << 1) + 1;
    // }

    inline int toMKey(int idx) {
        return idx;
    }

    inline void decodeParent(int parentCode, int& parentIdx, bool& fromRC) {
        parentIdx = parentCode >> 1;
        fromRC = parentCode & 1;
    }

    inline int transformParentCode(int parentCode, uint32_t globalOffset, int iBatch) {
        bool needTrans = iBatch > 0 && parentCode < 0;
        return needTrans ? -parentCode : (globalOffset << 1) + parentCode;
    }


    struct FNodesRawRepr {
        //uint32_t* __restrict_arr key;
        vec3f* __restrict_arr pt;
        uint8_t* __restrict_arr removal;
        int* __restrict_arr parentCode;

        inline void fromLeaf(const LeavesRawRepr& leaves, const vec3f* pts, uint32_t fi, uint32_t li, uint32_t globalOffset) {
            //key[fi] = toFKey(globalOffset + li);
            pt[fi] = pts[li];
            removal[fi] = leaves.replacedBy[li] == -1;
            parentCode[fi] = (globalOffset << 1) + leaves.parent[li];
        }
    };

    struct MNodesRawRepr {
        int* __restrict_arr key;
        int* __restrict_arr splitDim;
        mfloat* __restrict_arr splitVal;
        uint8_t* __restrict_arr removal;
        int* __restrict_arr parentCode;

        inline void fromInterior(const InteriorsRawRepr& interiors, uint32_t mi, uint32_t ii, uint32_t globalOffset, int iBatch) {
            key[mi] = toMKey(globalOffset + ii);
            splitDim[mi] = interiors.splitDim[ii];
            splitVal[mi] = interiors.splitVal[ii];
            removal[mi] = interiors.removeState[ii].load(std::memory_order_relaxed);
            parentCode[mi] = transformParentCode(interiors.parent[ii], globalOffset, iBatch);
        }
    };

    struct HNodesRawRepr {
        //uint32_t* __restrict_arr key;
        hash_t* __restrict_arr hash;
        int* __restrict_arr parentCode;

        inline void fromInterior(const InteriorsRawRepr& interiors, uint32_t hi, uint32_t ii, uint32_t globalOffset, int iBatch) {
            //key[hi] = toHKey(globalOffset + ii);
#ifdef ENABLE_MERKLE
            hash[hi] = interiors.hash[ii];
#endif
            parentCode[hi] = transformParentCode(interiors.parent[ii], globalOffset, iBatch);
        }

        inline void fromLeaf(const LeavesRawRepr& leaves, uint32_t hi, uint32_t li, uint32_t globalOffset) {
            //key[hi] = toFKey(globalOffset + li);
#ifdef ENABLE_MERKLE
            hash[hi] = leaves.hash[li];
#endif
            parentCode[hi] = (globalOffset << 1) + leaves.parent[li];
        }

        inline void fromNode(hash_t* _hash, int _parentCode, uint32_t globalOffset, int iBatch, int32_t hi) {
            //key[hi] = toKey(globalOffset + ni, isInterior);
#ifdef ENABLE_MERKLE
            hash[hi] = *_hash;
#endif
            parentCode[hi] = transformParentCode(_parentCode, globalOffset, iBatch);
        }
    };

    struct FNodes {
        //vector<int> key;
        vector<vec3f> pt;
        vector<uint8_t> removal;
        vector<int> parentCode;

        FNodes() = default;
        FNodes(FNodes&&) = default;
        FNodes& operator=(FNodes&&) = default;

        FNodes(const FNodes&) = delete;
        FNodes& operator=(const FNodes&) = delete;

        size_t size() const { return pt.size(); }

        void resize(size_t size) {
            //key.resize(size);
            pt.resize(size);
            removal.resize(size);
            parentCode.resize(size);
        }

        FNodes copyToHost() const {
            FNodes res;
            //res.key = key;
            res.pt = pt;
            res.removal = removal;
            res.parentCode = parentCode;
            return res;
        }

        FNodesRawRepr getRawRepr(size_t offset = 0u) {
            return FNodesRawRepr{
                //key.data() + offset,
                pt.data() + offset,
                removal.data() + offset,
                parentCode.data() + offset
            };
        }
    };

    struct MNodes {
        vector<int> key;
        vector<int> splitDim;
        vector<mfloat> splitVal;
        vector<uint8_t> removal;
        vector<int> parentCode;

        MNodes() = default;
        MNodes(MNodes&&) = default;
        MNodes& operator=(MNodes&&) = default;

        MNodes(const MNodes&) = delete;
        MNodes& operator=(const MNodes&) = delete;

        size_t size() const { return key.size(); }

        void resize(size_t size) {
            key.resize(size);
            splitDim.resize(size);
            splitVal.resize(size);
            removal.resize(size);
            parentCode.resize(size);
        }

        MNodes copyToHost() const {
            MNodes res;
            res.key = key;
            res.splitDim = splitDim;
            res.splitVal = splitVal;
            res.removal = removal;
            res.parentCode = parentCode;
            return res;
        }

        MNodesRawRepr getRawRepr(size_t offset = 0u) {
            return MNodesRawRepr{
                key.data() + offset,
                splitDim.data() + offset,
                splitVal.data() + offset,
                removal.data() + offset,
                parentCode.data() + offset
            };
        }
    };

    struct HNodes {
        //vector<int> key;
        vector<hash_t> hash;
        vector<int> parentCode;

        HNodes() = default;
        HNodes(HNodes&&) = default;
        HNodes& operator=(HNodes&&) = default;

        HNodes(const HNodes&) = delete;
        HNodes& operator=(const HNodes&) = delete;

        size_t size() const { return hash.size(); }

        void resize(size_t size) {
            //key.resize(size);
            hash.resize(size);
            parentCode.resize(size);
        }

        HNodes copyToHost() const {
            HNodes res;
            //res.key = key;
            res.hash = hash;
            res.parentCode = parentCode;
            return res;
        }

        HNodesRawRepr getRawRepr(size_t offset = 0u) {
            return HNodesRawRepr{
                //key.data() + offset,
                hash.data() + offset,
                parentCode.data() + offset
            };
        }
    };

    struct VerificationSet {
        FNodes fNodes;
        MNodes mNodes;
        HNodes hNodes;
    };
}