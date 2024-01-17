#pragma once
#include <parlay/parallel.h>
#include <parlay/slice.h>

namespace pmkd {
    // **************************************************************
    // An implementation of reduce.
    // Uses divide and conquer with a base case of block_size.
    // Works on arbitrary "ranges" (e.g. sequences, delayed sequences,
    //    std::vector, std::string, ..).
    // **************************************************************

    template<typename RetType, typename Range, typename BinaryOp>
    auto reduce(const Range& A, BinaryOp&& selfBinOp) {
        long n = A.size();
        long block_size = 100;
        if (n == 0) return RetType();
        if (n <= block_size) {
            auto v = RetType(A[0]);
            for (long i = 1; i < n; i++)
                selfBinOp(v, A[i]);
            return v;
        }

        RetType L, R;
        parlay::par_do([&] {L = reduce<RetType>(parlay::make_slice(A).cut(0, n / 2), selfBinOp);},
            [&] {R = reduce<RetType>(parlay::make_slice(A).cut(n / 2, n), selfBinOp);});
        selfBinOp(L, R);
        return L;
    }
}
