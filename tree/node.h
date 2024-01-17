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
		int* __restrict__ primIdx;
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
		vector<int> primIdx;
		vector<int> segOffset;
		vector<MortonType> morton;

		LeavesRawRepr getRawRepr(size_t offset = 0u) {
			return LeavesRawRepr{
				primIdx.data() + offset,
				segOffset.data() + offset,
				morton.data() + offset };
		}

		LeavesRawRepr getRawRepr(size_t offset = 0u) const {
			return LeavesRawRepr{
				const_cast<int*>(primIdx.data()) + offset,
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