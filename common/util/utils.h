#pragma once
#include <fmt/printf.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <parlay/parallel.h>
#include <parlay/slice.h>

namespace pmkd {
#ifdef M_DEBUG
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

    template<typename F, typename ...Args>
    void mTimer(const std::string& msg, F&& func, Args&&...args) {
        auto start = std::chrono::high_resolution_clock::now();
        func(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end - start;
        fmt::print("{}: {}ms\n", msg, elapsed_time.count() * 1000.0);
    }

    template<typename F, typename ...Args>
    void mTimerRepeated(const std::string& msg, int nIter, F&& func, Args&&...args) {
        double avgTime = 0.0;
        for (int i = 0; i < nIter;++i) {
            auto start = std::chrono::high_resolution_clock::now();
            func(std::forward<Args>(args)...);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = end - start;
            avgTime += elapsed_time.count();
        }
        fmt::print("{}: {}ms\n", msg, avgTime * 1000.0 / (double)nIter);
    }
}
