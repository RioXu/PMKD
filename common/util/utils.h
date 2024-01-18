#pragma once
#include <parlay/parallel.h>
#include <parlay/slice.h>

namespace pmkd {
#ifdef DEBUG
#define MASSERT(expr) assert(expr)
#else
#define MASSERT(expr) 
#endif


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

    template<typename T>
    struct VecHash {
        std::size_t operator()(const T& p) const {
            std::size_t result = std::hash<typename T::value_type>{}(p[0]);
            for (int i = 1; i < p.size(); i++) {
                result ^= (std::hash<typename T::value_type>{}(p[i]) << i);
            }
            return result;
        }
    };
}
