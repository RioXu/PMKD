#pragma once
#include <atomic>
#include <parlay/sequence.h>
#include "morton.h"

namespace pmkd {
	
	using MortonType = Morton<64>;
	using TreeIdx = uint32_t;

	template<typename T>
	using vector = parlay::sequence<T>;

	// using Structure of Arrays (SOA) pattern
	struct LeavesRawRepr {
		//int* parent;
		//int* __restrict__ primIdx;
		//int* segLen;
		int* __restrict__ segOffset;
		//TreeIdx* escape;
		//int* parentSplitDim;
		//mfloat* parentSplitVal;
		MortonType* __restrict__ morton;
	};

	struct InteriorsRawRepr {
		//int* parent;
		//TreeIdx* lc, * rc;
		int* __restrict__ rangeL;
		int* __restrict__ rangeR;
		//TreeIdx* escape;
		int* __restrict__ splitDim;
		mfloat* __restrict__ splitVal;
		int* __restrict__ parentSplitDim;
		mfloat* __restrict__ parentSplitVal;
	};

	struct Leaves {
		//vector<int> primIdx;
		vector<int> segOffset;
		vector<MortonType> morton;

		Leaves() = default;
		Leaves(Leaves&&) = default;
		Leaves& operator=(Leaves&&) = default;

		Leaves(const Leaves&) = delete;
		Leaves& operator=(const Leaves&) = delete;

		size_t size() const { return segOffset.size(); }

		void reserve(size_t capacity) {
			//primIdx.reserve(capacity);
			segOffset.reserve(capacity);
			morton.reserve(capacity);
		}

		void resize(size_t size) {
			//primIdx.resize(size);
			segOffset.resize(size);
			morton.resize(size);
		}

		LeavesRawRepr getRawRepr(size_t offset = 0u) {
			return LeavesRawRepr{
				//primIdx.data() + offset,
				segOffset.data() + offset,
				morton.data() + offset };
		}

		LeavesRawRepr getRawRepr(size_t offset = 0u) const {
			return LeavesRawRepr{
				//const_cast<int*>(primIdx.data()) + offset,
				const_cast<int*>(segOffset.data()) + offset,
				const_cast<MortonType*>(morton.data()) + offset };
		}
	};

	struct Interiors {
		vector<int> rangeL, rangeR;
		vector<int> splitDim;
		vector<mfloat> splitVal;
		vector<int> parentSplitDim;
		vector<mfloat> parentSplitVal;

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
		}

		void resize(size_t size) {
			rangeL.resize(size);
			rangeR.resize(size);
			splitDim.resize(size);
			splitVal.resize(size);
			parentSplitDim.resize(size);
			parentSplitVal.resize(size);
		}

		InteriorsRawRepr getRawRepr(size_t offset = 0u) {
			return InteriorsRawRepr{
				rangeL.data()+offset, rangeR.data()+offset,
				splitDim.data()+offset,splitVal.data()+offset,
				parentSplitDim.data()+offset,parentSplitVal.data()+offset
			};
		}

		InteriorsRawRepr getRawRepr(size_t offset = 0u) const {
			return InteriorsRawRepr{
				const_cast<int*>(rangeL.data())+offset, const_cast<int*>(rangeR.data())+offset,
				const_cast<int*>(splitDim.data())+offset,const_cast<mfloat*>(splitVal.data())+offset,
				const_cast<int*>(parentSplitDim.data())+offset,const_cast<mfloat*>(parentSplitVal.data())+offset
			};
		}
	};

	struct NodeMgrDevice {
		// stored on device
		size_t numBatches = 0;
		LeavesRawRepr* leavesBatch = nullptr;  // note: the array can be stored in GPU constant memory
		InteriorsRawRepr* interiorsBatch = nullptr;
		int* sizesAcc = nullptr;
	};

	void transformLeafIdx(int idx, int* sizesAcc, size_t numBatches, int& iBatch, int& offset) {
		iBatch = *(std::upper_bound(sizesAcc, sizesAcc + numBatches, idx) - 1);
		offset = idx - sizesAcc[iBatch];
	}

	class NodeMgr {
	private:
		// stored on host
		vector<Leaves> leavesBatch;
		vector<Interiors> interiorsBatch;
		vector<int> sizesAcc; // prefix sum of the sizes of each leaf batch

		// stored on device
		vector<LeavesRawRepr> dLeavesBatch;
		vector<InteriorsRawRepr> dInteriorsBatch;
		vector<int> dSizesAcc;

		void clearHost() {
			leavesBatch.clear();
            interiorsBatch.clear();
            sizesAcc.clear();
		}

		void clearDevice() {
            dLeavesBatch.clear();
            dInteriorsBatch.clear();
            dSizesAcc.clear();
        }
	public:
		size_t numBatches() const { return leavesBatch.size(); }

		void append(Leaves&& leaves, Interiors&& interiors, bool syncDevice = true) {
			if (interiors.rangeL.empty()) return;

			int acc = numBatches() == 0 ? 0 : sizesAcc[numBatches() - 1] + leavesBatch[numBatches() - 1].size();

			if (syncDevice) {
				dSizesAcc.push_back(acc);
				dLeavesBatch.push_back(leaves.getRawRepr());
				dInteriorsBatch.push_back(interiors.getRawRepr());
			}

			sizesAcc.push_back(acc);
			leavesBatch.emplace_back(std::move(leaves));
			interiorsBatch.emplace_back(std::move(interiors));
		}

		void clear() {
			clearHost();
            clearDevice();
		}

		Leaves& getLeaves(size_t batchIdx) { return leavesBatch[batchIdx]; }

		Interiors& getInteriors(size_t batchIdx) { return interiorsBatch[batchIdx]; }

		bool isDeviceSyncronized() const { return dLeavesBatch.size() == leavesBatch.size(); }

		void syncDevice() {
			if (isDeviceSyncronized()) return;
			if (numBatches() == 0) {
				clearDevice();
				return;
			}
			// note: resize can be async
			dLeavesBatch.resize(leavesBatch.size());
            dInteriorsBatch.resize(interiorsBatch.size());
            dSizesAcc.resize(sizesAcc.size());

			vector<LeavesRawRepr> hLeavesBatch;
			for (size_t i = 0; i < leavesBatch.size(); i++) {
				hLeavesBatch.push_back(leavesBatch[i].getRawRepr());
			}
			vector<InteriorsRawRepr> hInteriorsBatch;
			for (size_t i = 0; i < interiorsBatch.size(); i++) {
				hInteriorsBatch.push_back(interiorsBatch[i].getRawRepr());
			}
			memcpy(dLeavesBatch.data(), hLeavesBatch.data(), sizeof(LeavesRawRepr) * leavesBatch.size());
			memcpy(dInteriorsBatch.data(), hInteriorsBatch.data(), sizeof(InteriorsRawRepr) * interiorsBatch.size());
			memcpy(dSizesAcc.data(), sizesAcc.data(), sizeof(int) * sizesAcc.size());
		}

		NodeMgrDevice getDeviceRepr() {
			syncDevice();

			NodeMgrDevice dNodeMgr;
			dNodeMgr.numBatches = leavesBatch.size();
			dNodeMgr.leavesBatch = dLeavesBatch.data();
			dNodeMgr.interiorsBatch = dInteriorsBatch.data();
			dNodeMgr.sizesAcc = dSizesAcc.data();
			return dNodeMgr;
		}
	};

	struct AtomicCount {
		std::atomic<int> cnt;  // size is 4
		//char pack[60];       // note: this optimization may not be necessary
	}; 

	struct BuildAid {
		int* __restrict__ metrics;
		AtomicCount* visitCount;
		int* __restrict__ leftLeafCount;
		int* __restrict__ segLen;
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