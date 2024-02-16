#pragma once
#include <immintrin.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

namespace pmkd {
    void compare_vectors(const float* vec1, const float* vec2, int* result);

    template<typename T>
    inline decltype(auto) fromConstPtr(const T* ptr) {
        return const_cast<T*>(ptr);
    }

    constexpr const size_t _binary_search_base = 16;

    template <typename T, typename F>
    size_t linear_search(T* const I, size_t size, const T& v,
        const F& less) {
        for (size_t i = 0; i < size; i++)
            if (!less(I[i], v)) return i;
        return size;
    }

    // return index to first key greater or equal to v
    template <typename T, typename F>
    size_t binary_search(T* const I, size_t size, const T& v,
        const F& less) {
        size_t start = 0;
        size_t end = size;
        while (end - start > _binary_search_base) {
            size_t mid = (end + start) / 2;
            if (!less(I[mid], v))
                end = mid;
            else
                start = mid + 1;
        }
        return start + linear_search(I + start, end - start, v, less);
    }

    // merge addIdx and leafIdx into oPrimIdx
    // merge leafIdx and binIdx into oBinIdx
    template <typename T, typename BinaryOp>
    void seq_merge(T* addIdx, T* addIdxEnd, T* leafIdx, T* leafIdxEnd, T* binIdx, T* oPrimIdx, T* oBinIdx, BinaryOp&& f) {
        size_t nA = addIdxEnd - addIdx;  // nA = addIdx.size() = binIdx.size()
        size_t nB = leafIdxEnd - leafIdx;
        size_t i = 0;
        size_t j = 0;
        while (true) {
            if (i == nA) {
                while (j < nB) {
                    oPrimIdx[i + j] = leafIdx[j];
                    oBinIdx[i + j] = leafIdx[j];
                    j++;
                }
                break;
            }
            if (j == nB) {
                while (i < nA) {
                    oPrimIdx[i + j] = addIdx[i];
                    oBinIdx[i + j] = binIdx[i];
                    i++;
                }
                break;
            }
            if (f(leafIdx[j], addIdx[i])) {
                oPrimIdx[i + j] = leafIdx[j];
                oBinIdx[i + j] = leafIdx[j];
                j++;
            }
            else {
                oPrimIdx[i + j] = addIdx[i];
                oBinIdx[i + j] = binIdx[i];
                i++;
            }
        }
    }


    template <typename T, typename BinaryOp>
    void _mergeZip(T* addIdx, T* addIdxEnd, T* leafIdx, T* leafIdxEnd, T* binIdx, T* oPrimIdx, T* oBinIdx, BinaryOp&& f) {

        size_t nA = addIdxEnd - addIdx; // nA = addIdx.size() = binIdx.size()
        size_t nB = leafIdxEnd - leafIdx;
        size_t nR = nA + nB;
        if (nR < parlay::internal::_merge_base) {
            seq_merge(addIdx, addIdxEnd, leafIdx, leafIdxEnd, binIdx, oPrimIdx, oBinIdx, f);
        }
        else if (nA == 0) {
            parlay::parallel_for(0, nB, [&](size_t i) {
                oPrimIdx[i] = leafIdx[i];
                oBinIdx[i] = leafIdx[i];
                });
        }
        else if (nB == 0) {
            parlay::parallel_for(0, nA, [&](size_t i) {
                oPrimIdx[i] = addIdx[i];
                oBinIdx[i] = binIdx[i];
                });
        }
        else {
            size_t mA = nA / 2;
            // important for stability that binary search identifies
            // first element in B greater or equal to A[mA]
            size_t mB = binary_search(leafIdx, nB, addIdx[mA], f);
            if (mB == 0) mA++;  // ensures at least one on each side
            size_t mR = mA + mB;
            auto left = [&]() {
                _mergeZip(addIdx, addIdx + mA, leafIdx, leafIdx + mB, binIdx,
                    oPrimIdx, oBinIdx, f);
                };
            auto right = [&]() {
                _mergeZip(addIdx + mA, addIdx + nA, leafIdx + mB, leafIdx + nB, binIdx + mA,
                    oPrimIdx + mR, oBinIdx + mR, f);
                };
            parlay::par_do(left, right, false);
        }
    }

    // merge addIdx and leafIdx into oPrimIdx
    // merge leafIdx and binIdx into oBinIdx
    template <typename R1, typename R2, typename BinaryOp>
    void mergeZip(R1&& addIdx, R2&& leafIdx, R1&& binIdx, R1&& oPrimIdx, R1&& oBinIdx, BinaryOp&& f) {
        _mergeZip(
            addIdx.data(), addIdx.data() + addIdx.size(),
            leafIdx.data(), leafIdx.data() + leafIdx.size(),
            binIdx.data(), oPrimIdx.data(), oBinIdx.data(), f
        );
    }
}