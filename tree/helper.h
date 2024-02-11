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

    // merge addIdx and leafIdx into oPrimIdx
    // merge leafIdx and binIdx into oBinIdx
    template <typename R, typename BinaryOp>
    void seq_merge(R&& addIdx, R&& leafIdx, R&& binIdx, R&& oPrimIdx, R&& oBinIdx, BinaryOp&& f) {
        size_t nA = addIdx.size();  // nA = addIdx.size() = binIdx.size()
        size_t nB = leafIdx.size();
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

    // merge addIdx and leafIdx into oPrimIdx
    // merge leafIdx and binIdx into oBinIdx
    template <typename R, typename BinaryOp>
    void mergeZip(R&& addIdx, R&& leafIdx, R&& binIdx, R&& oPrimIdx, R&& oBinIdx, BinaryOp&& f) {

        size_t nA = addIdx.size(); // nA = addIdx.size() = binIdx.size()
        size_t nB = leafIdx.size();
        size_t nR = nA + nB;
        if (nR < parlay::internal::_merge_base) {
            seq_merge(addIdx, leafIdx, binIdx, oPrimIdx, oBinIdx, f);
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
            size_t mB = parlay::internal::binary_search(leafIdx, addIdx[mA], f);
            if (mB == 0) mA++;  // ensures at least one on each side
            size_t mR = mA + mB;
            auto left = [&]() {
                mergeZip(addIdx.cut(0, mA), leafIdx.cut(0, mB), binIdx.cut(0, mA),
                    oPrimIdx.cut(0, mR), oBinIdx.cut(0, mR), f);
                };
            auto right = [&]() {
                mergeZip(addIdx.cut(mA, nA), leafIdx.cut(mB, nB), binIdx.cut(mA, nA),
                    oPrimIdx.cut(mR, nR), oBinIdx.cut(mR, nR), f);
                };
            parlay::par_do(left, right, false);
        }
    }
}