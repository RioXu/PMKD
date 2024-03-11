#pragma once
#include <atomic>
#include <parlay/sequence.h>
#include <vector>

#include <morton.h>
#include <auth/sha.h>


namespace pmkd {
	
	using MortonType = Morton<64>;
	using TreeIdx = uint32_t;

	template<typename T>
	//using vector = parlay::sequence<T>;
	using vector = std::vector<T>;

	struct AtomicCount {
		std::atomic<uint8_t> cnt;  // size is 1
		//char pack[63];       // note: this optimization may not be necessary

		AtomicCount() : cnt(0) {}
	};
	// using atomic_t = std::atomic<int>;
	using atomic_t = std::atomic_uint8_t;
	using hash_t = sha256_t;


	using BottomUpState = atomic_t;

	struct TopDownStates {
		uint8_t* __restrict_arr child[2];
	};

	// using Structure of Arrays (SOA) pattern
	struct LeavesRawRepr {
		int* __restrict_arr segOffset;
		MortonType* __restrict_arr morton;
		int* __restrict_arr parent;
		// for dynamic tree
		int* __restrict_arr treeLocalRangeR;
		int* __restrict_arr replacedBy;
		int* __restrict_arr derivedFrom;
#ifdef ENABLE_MERKLE
		hash_t* __restrict_arr hash;
#endif
	};

	struct InteriorsRawRepr {
		int* __restrict_arr rangeL;
		int* __restrict_arr rangeR;
		int* __restrict_arr splitDim;
		mfloat* __restrict_arr splitVal;
		int* __restrict_arr parent;
		// for dynamic tree
		BottomUpState* __restrict_arr removeState;
#ifdef ENABLE_MERKLE
		BottomUpState* __restrict_arr visitState;
		TopDownStates visitStateTopDown;
		hash_t* __restrict_arr hash;
#endif
	};

	struct Leaves {
		//vector<int> primIdx;
		parlay::sequence<int> segOffset;
		vector<MortonType> morton;
		vector<int> parent;
		// for dynamic tree
		vector<int> treeLocalRangeR;  // exclusive, i.e. [L, R)
		vector<int> replacedBy; // 0: not replaced, -1: removed, positive: replaced
		vector<int> derivedFrom;
#ifdef ENABLE_MERKLE
		vector<hash_t> hash;
#endif

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
			parent.reserve(capacity);

			treeLocalRangeR.reserve(capacity);
			replacedBy.reserve(capacity);
			derivedFrom.reserve(capacity);
#ifdef ENABLE_MERKLE
			hash.reserve(capacity);
#endif
		}

		void resizePartial(size_t size) {
			morton.resize(size);
			replacedBy.resize(size, 0);
			parent.resize(size);
#ifdef ENABLE_MERKLE
			hash.resize(size);
#endif
		}

		void resize(size_t size) {
			resizePartial(size);

			segOffset.resize(size);
			treeLocalRangeR.resize(size);
			derivedFrom.resize(size);
		}

		Leaves copyToHost() const {
			Leaves res;
            res.segOffset = segOffset;
			res.morton = morton;
			res.parent = parent;
			res.treeLocalRangeR = treeLocalRangeR;
            res.replacedBy = replacedBy;
			res.derivedFrom = derivedFrom;
#ifdef ENABLE_MERKLE
			res.hash = hash;
#endif
			return res;
		}

		LeavesRawRepr getRawRepr(size_t offset = 0u) {
			return LeavesRawRepr{
				segOffset.data() + offset,
				morton.data() + offset,
				parent.data() + offset,
				treeLocalRangeR.data() + offset,
				replacedBy.data() + offset,
				derivedFrom.data() + offset,
				#ifdef ENABLE_MERKLE
				hash.data() + offset,
                #endif
			};
		}

		LeavesRawRepr getRawRepr(size_t offset = 0u) const {
			return LeavesRawRepr{
				const_cast<int*>(segOffset.data()) + offset,
				const_cast<MortonType*>(morton.data()) + offset,
				const_cast<int*>(parent.data()) + offset,
				const_cast<int*>(treeLocalRangeR.data()) + offset,
				const_cast<int*>(replacedBy.data()) + offset,
				const_cast<int*>(derivedFrom.data()) + offset,
				#ifdef ENABLE_MERKLE
				const_cast<hash_t*>(hash.data()) + offset,
                #endif
			};
		}
	};

	struct Interiors {
		vector<int> rangeL, rangeR;
		vector<int> splitDim;
		vector<mfloat> splitVal;
		vector<int> parent;
		// for dynamic tree
		// remove states
		// 01b: lc removed, 10b: rc removed, 11b: both removed
		vector<BottomUpState> removeState;
#ifdef ENABLE_MERKLE
		vector<BottomUpState> visitState;  // make sure is cleared before use
		vector<uint8_t> vsLeftChild;
		vector<uint8_t> vsRightChild;
		vector<hash_t> hash;
#endif

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
			parent.reserve(capacity);

#ifdef ENABLE_MERKLE
			vsLeftChild.reserve(capacity);
			vsRightChild.reserve(capacity);
			hash.reserve(capacity);
#endif
		}

		void resize(size_t size) {
			rangeL.resize(size);
			rangeR.resize(size);
			splitDim.resize(size);
			splitVal.resize(size);
			parent.resize(size);
			
			removeState = vector<BottomUpState>(size);
#ifdef ENABLE_MERKLE
			visitState = vector<BottomUpState>(size);
			vsLeftChild.clear();
			vsLeftChild.resize(size, 0);
			vsRightChild.clear();
			vsRightChild.resize(size, 0);
			hash.resize(size);
#endif
		}

		Interiors copyToHost() const {
			Interiors res;
            res.rangeL = rangeL;
            res.rangeR = rangeR;
            res.splitDim = splitDim;
            res.splitVal = splitVal;
            res.parent = parent;

			res.removeState = vector<BottomUpState>(removeState.size());
			for (size_t i = 0; i < removeState.size(); ++i) {
				res.removeState[i] = removeState[i].load(std::memory_order_relaxed);
			}
#ifdef ENABLE_MERKLE
			res.hash = hash;
#endif
			return res;
		}

		InteriorsRawRepr getRawRepr(size_t offset = 0u) {
			return InteriorsRawRepr{
				rangeL.data() + offset, rangeR.data() + offset,
				splitDim.data() + offset,splitVal.data() + offset,
				parent.data() + offset,
				removeState.data() + offset,
				#ifdef ENABLE_MERKLE
				visitState.data() + offset,
				{vsLeftChild.data() + offset, vsRightChild.data() + offset},
				hash.data() + offset
				#endif
			};
		}

		InteriorsRawRepr getRawRepr(size_t offset = 0u) const {
			return InteriorsRawRepr{
				const_cast<int*>(rangeL.data())+offset, const_cast<int*>(rangeR.data())+offset,
				const_cast<int*>(splitDim.data())+offset,const_cast<mfloat*>(splitVal.data())+offset,
				const_cast<int*>(parent.data()) + offset,
				const_cast<BottomUpState*>(removeState.data()) + offset,
				#ifdef ENABLE_MERKLE
				const_cast<BottomUpState*>(visitState.data()) + offset,
				{
					const_cast<uint8_t*>(vsLeftChild.data()) + offset,
				    const_cast<uint8_t*>(vsRightChild.data()) + offset
				},
				const_cast<hash_t*>(hash.data()) + offset
				#endif
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
		uint8_t* __restrict_arr metrics;
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