#pragma once
#include <atomic>
#include <parlay/sequence.h>
#include <vector>
#include "morton.h"

namespace pmkd {
	
	using MortonType = Morton<64>;
	using TreeIdx = uint32_t;

	template<typename T>
	//using vector = parlay::sequence<T>;
	using vector = std::vector<T>;

	struct AtomicCount {
		std::atomic<int> cnt;  // size is 4
		//char pack[60];       // note: this optimization may not be necessary

		AtomicCount() : cnt(0) {}
	};

	// using Structure of Arrays (SOA) pattern
	struct LeavesRawRepr {
		int* __restrict_arr segOffset;
		MortonType* __restrict_arr morton;
		// for dynamic tree
		int* __restrict_arr treeLocalRangeR;
		int* __restrict_arr replacedBy;
		int* __restrict_arr derivedFrom;
	};

	struct InteriorsRawRepr {
		int* __restrict_arr rangeL;
		int* __restrict_arr rangeR;
		int* __restrict_arr splitDim;
		mfloat* __restrict_arr splitVal;
		int* __restrict_arr parentSplitDim;
		mfloat* __restrict_arr parentSplitVal;
		// for dynamic tree
		uint8_t* __restrict_arr metrics;
		int* __restrict_arr mapidx;
		AtomicCount* __restrict_arr alterState;
	};

	struct Leaves {
		//vector<int> primIdx;
		parlay::sequence<int> segOffset;
		vector<MortonType> morton;
		// for dynamic tree
		vector<int> treeLocalRangeR;  // exclusive, i.e. [L, R)
		vector<int> replacedBy; // 0: not replaced, -1: removed, positive: replaced
		vector<int> derivedFrom;

		Leaves() = default;
		Leaves(Leaves&&) = default;
		Leaves& operator=(Leaves&&) = default;

		Leaves(const Leaves&) = delete;
		Leaves& operator=(const Leaves&) = delete;

		size_t size() const { return morton.size(); }

		void reserve(size_t capacity) {
			//primIdx.reserve(capacity);
			segOffset.reserve(capacity);
			morton.reserve(capacity);
			treeLocalRangeR.reserve(capacity);
			replacedBy.reserve(capacity);
			derivedFrom.reserve(capacity);
		}

		void resize(size_t size) {
			//primIdx.resize(size);
			segOffset.resize(size);
			morton.resize(size);
			treeLocalRangeR.resize(size);
			replacedBy.resize(size);
			derivedFrom.resize(size);
		}

		// void append(const Leaves& other) {
        //     segOffset.append(other.segOffset);
        //     morton.append(other.morton);
        //     treeLocalRangeR.append(other.treeLocalRangeR);
        //     replacedBy.append(other.replacedBy);
        //     derivedFrom.append(other.derivedFrom);
		// }

		Leaves copyToHost() const {
			Leaves res;
            res.segOffset = segOffset;
            res.morton = morton;
            res.treeLocalRangeR = treeLocalRangeR;
            res.replacedBy = replacedBy;
            res.derivedFrom = derivedFrom;
            return res;
		}

		LeavesRawRepr getRawRepr(size_t offset = 0u) {
			return LeavesRawRepr{
				segOffset.data() + offset,
				morton.data() + offset,
				treeLocalRangeR.data() + offset,
				replacedBy.data() + offset,
				derivedFrom.data() + offset };
		}

		LeavesRawRepr getRawRepr(size_t offset = 0u) const {
			return LeavesRawRepr{
				const_cast<int*>(segOffset.data()) + offset,
				const_cast<MortonType*>(morton.data()) + offset,
				const_cast<int*>(treeLocalRangeR.data()) + offset,
				const_cast<int*>(replacedBy.data()) + offset,
				const_cast<int*>(derivedFrom.data()) + offset };
		}
	};

	struct Interiors {
		vector<int> rangeL, rangeR;
		vector<int> splitDim;
		vector<mfloat> splitVal;
		vector<int> parentSplitDim;
		vector<mfloat> parentSplitVal;
		// for dynamic tree
		vector<uint8_t> metrics;
		vector<int> mapidx;   // original index to optimized layout index
		// 1b: lc visit, 10b: rc visit, 100b: lc removal, 1000b: rc removal
		vector<AtomicCount> alterState;

		Interiors() = default;
		Interiors(Interiors&&) = default;
		Interiors& operator=(Interiors&&) = default;

		Interiors(const Interiors&) = delete;
		Interiors& operator=(const Interiors&) = delete;

		size_t size() const { return rangeL.size(); }

		void reserve(size_t capacity) {
			rangeL.reserve(capacity);
			rangeR.reserve(capacity);
			splitDim.reserve(capacity);
			splitVal.reserve(capacity);
			parentSplitDim.reserve(capacity);
			parentSplitVal.reserve(capacity);

			metrics.reserve(capacity);
			mapidx.reserve(capacity);
			//alterState.reserve(capacity);
		}

		void resize(size_t size) {
			rangeL.resize(size);
			rangeR.resize(size);
			splitDim.resize(size);
			splitVal.resize(size);
			parentSplitDim.resize(size);
			parentSplitVal.resize(size);

			metrics.resize(size);
            mapidx.resize(size);
            alterState = vector<AtomicCount>(size);
		}

		// void append(const Interiors& other) {
		// 	rangeL.append(other.rangeL);
        //     rangeR.append(other.rangeR);
        //     splitDim.append(other.splitDim);
        //     splitVal.append(other.splitVal);
        //     parentSplitDim.append(other.parentSplitDim);
		// 	parentSplitVal.append(other.parentSplitVal);

		// 	metrics.append(other.metrics);
        //     mapidx.append(other.mapidx);
        //     //alterState.append(other.alterState);
		// }

		Interiors copyToHost() const {
			Interiors res;
            res.rangeL = rangeL;
            res.rangeR = rangeR;
            res.splitDim = splitDim;
            res.splitVal = splitVal;
            res.parentSplitDim = parentSplitDim;
			res.parentSplitVal = parentSplitVal;

			res.metrics = metrics;
			res.mapidx = mapidx;
			res.alterState = vector<AtomicCount>(alterState.size());
			for (size_t i = 0; i < alterState.size(); ++i) {
                res.alterState[i].cnt = alterState[i].cnt.load();
            }
			return res;
		}

		InteriorsRawRepr getRawRepr(size_t offset = 0u) {
			return InteriorsRawRepr{
				rangeL.data() + offset, rangeR.data() + offset,
				splitDim.data() + offset,splitVal.data() + offset,
				parentSplitDim.data() + offset,parentSplitVal.data() + offset,
				metrics.data() + offset, mapidx.data() + offset,
				alterState.data() + offset
			};
		}

		InteriorsRawRepr getRawRepr(size_t offset = 0u) const {
			return InteriorsRawRepr{
				const_cast<int*>(rangeL.data())+offset, const_cast<int*>(rangeR.data())+offset,
				const_cast<int*>(splitDim.data())+offset,const_cast<mfloat*>(splitVal.data())+offset,
				const_cast<int*>(parentSplitDim.data()) + offset,const_cast<mfloat*>(parentSplitVal.data()) + offset,
				const_cast<uint8_t*>(metrics.data())+offset, const_cast<int*>(mapidx.data())+offset,
                const_cast<AtomicCount*>(alterState.data())+offset
			};
		}
	};

	struct NodeMgrDevice {
		// stored on device
		size_t numBatches = 0;
		LeavesRawRepr* leavesBatch = nullptr;  // note: the array can be stored in GPU constant memory
		InteriorsRawRepr* interiorsBatch = nullptr;
		vec3f** ptsBatch = nullptr;
		int* sizesAcc = nullptr;
	};

	inline void transformLeafIdx(int globalIdx, int* sizesAcc, size_t numBatches, int& iBatch, int& offset) {
		iBatch = std::upper_bound(sizesAcc, sizesAcc + numBatches, globalIdx) - sizesAcc;
		offset = globalIdx - (iBatch > 0 ? sizesAcc[iBatch-1] : 0);
	}

	class NodeMgr {
	private:
		// handles stored on host, data stored on device
		vector<Leaves> leavesBatch;
		vector<Interiors> interiorsBatch;
		vector<vector<vec3f>> ptsBatch;
		vector<int> sizesAcc; // inclusive prefix sum of the sizes of each leaf batch

		// handles stored on device
		vector<LeavesRawRepr> dLeavesBatch;
		vector<InteriorsRawRepr> dInteriorsBatch;
		vector<vec3f*> dPtsBatch;
		vector<int> dSizesAcc;

		void clearHost() {
			leavesBatch.clear();
			interiorsBatch.clear();
			ptsBatch.clear();
			sizesAcc.clear();
		}

		void clearDevice() {
            dLeavesBatch.clear();
			dInteriorsBatch.clear();
			dPtsBatch.clear();
			dSizesAcc.clear();
        }
	public:
		size_t numBatches() const { return leavesBatch.size(); }

		size_t numLeaves() const {
			size_t nB = numBatches();
			return nB == 0 ? 0 : sizesAcc[nB - 1];
		}

		void append(Leaves&& leaves, Interiors&& interiors, vector<vec3f>&& pts, bool syncDevice = true);

		void clear() {
			clearHost();
            clearDevice();
		}

		const Leaves& getLeaves(size_t batchIdx) const { return leavesBatch[batchIdx]; }
		Leaves& getLeaves(size_t batchIdx) { return leavesBatch[batchIdx]; }

		const Interiors& getInteriors(size_t batchIdx) const { return interiorsBatch[batchIdx]; }
		Interiors& getInteriors(size_t batchIdx) { return interiorsBatch[batchIdx]; }

		const vector<vec3f>& getPtsBatch(size_t batchIdx) const { return ptsBatch[batchIdx]; }
		vector<vec3f>& getPtsBatch(size_t batchIdx) { return ptsBatch[batchIdx]; }

		vector<vec3f> flattenPoints() const;

		bool isDeviceSyncronized() const { return dLeavesBatch.size() == leavesBatch.size(); } // note: check when tree remove function is implemented

		void syncDevice();

		NodeMgrDevice getDeviceHandle() const;

		struct HostCopy {
			vector<Leaves> leavesBatch;
			vector<Interiors> interiorsBatch;
			vector<vector<vec3f>> ptsBatch;
			vector<int> sizesAcc;
		};

		HostCopy copyToHost() const;
	};

	struct BuildAid {
		//uint8_t* __restrict_arr metrics;
		AtomicCount* visitCount;
		int* __restrict_arr leftLeafCount;
		int* __restrict_arr segLen;
	};

	inline int isLeaf(const TreeIdx idx, int* idxReal) {
		*idxReal = idx >> 1;
		return idx & 1;
	}

	inline TreeIdx toInteriorIdx(const int idx) {
		return idx << 1;
	}

	inline TreeIdx toLeafIdx(const int idx) {
		return (idx << 1) + 1;
	}
}