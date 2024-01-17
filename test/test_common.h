#pragma once
#include <fmt/printf.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <fmtlog.h>
#include <iostream>
#include <bitset>
#include <vector>
#include <unordered_set>
#include <functional>
#include <random>
#include <chrono>
#include <memory>
#include <fstream>
#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include <common/geometry/aabb.h>
#include <common/util/par_algo.h>
#include <pm_kdtree.h>



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

parlay::sequence<pmkd::vec3f> genPts(size_t n, bool random = true, bool printInfo = true) {
    parlay::sequence<pmkd::vec3f> points;
    points.reserve(n);

    unsigned seed = random ? std::chrono::system_clock::now().time_since_epoch().count() : 0;
    std::mt19937 gen(seed); // 以随机数种子初始化 Mersenne Twister 伪随机数生成器  
    std::normal_distribution<pmkd::mfloat> dis(0.0, 5.0);

    // 生成n个随机点  
    for (int i = 0; i < n; ++i) {
        points.emplace_back(dis(gen), dis(gen), dis(gen));
    }
    if (printInfo) {
        for (const auto& pt : points) {
            fmt::printf("(%.3f, %.3f, %.3f) ", pt.x, pt.y, pt.z);
        }
        fmt::printf("\n");
    }
    return points;
}

parlay::sequence<pmkd::AABB> genRanges(size_t n, bool random = true, bool printInfo = true) {
    parlay::sequence<pmkd::AABB> ranges;
    ranges.reserve(n);

    unsigned seed = random ? std::chrono::system_clock::now().time_since_epoch().count() : 0;
    std::mt19937 gen(seed); // 以随机数种子初始化 Mersenne Twister 伪随机数生成器  
    std::normal_distribution<pmkd::mfloat> dis1(0.0, 5.0);
    std::uniform_real_distribution<pmkd::mfloat> dis2(0.1, 3.0);

    // 生成n个随机AABB
    for (int i = 0; i < n; ++i) {
        pmkd::vec3f center(dis1(gen), dis1(gen), dis1(gen));
        pmkd::vec3f lengthVec(dis2(gen), dis2(gen), dis2(gen));
        ranges.emplace_back(center - lengthVec, center + lengthVec);
    }
    if (printInfo) {
        for (const auto& range : ranges) {
            fmt::print("{} ", range.toString());
        }
        fmt::printf("\n");
    }
    return ranges;
}

pmkd::RangeQueryResponses rangeQuery_Brutal(
    const parlay::sequence<pmkd::RangeQuery>& queries, const parlay::sequence<pmkd::vec3f>& pts) {
    if (queries.empty()) return pmkd::RangeQueryResponses(0);

    size_t nq = queries.size();
    pmkd::RangeQueryResponses responses(nq);

    for (size_t i = 0; i < nq; ++i) {
        auto resp = responses.at(i);

        const auto& query = queries[i];
        for (const auto& pt : pts) {
            if (query.include(pt)) {
                auto& respSize = *(resp.size);
                resp.pts[respSize++] = pt;
                if (respSize == responses.capPerResponse) break;
            }
        }
    }
    return responses;
}

bool isContentEqual(const pmkd::RangeQueryResponse& a, const pmkd::RangeQueryResponse& b) {
    bool isNull_a = !a.pts && !a.size;
    bool isNull_b = !b.pts && !b.size;
    if (isNull_a && isNull_b) return true;
    if ((isNull_a && !isNull_b) || (!isNull_a && isNull_b)) return false;
    // both are not null
    if (*a.size != *b.size) return false;

    std::unordered_set<pmkd::vec3f, VecHash<pmkd::vec3f>> set_a;
    std::unordered_set<pmkd::vec3f, VecHash<pmkd::vec3f>> set_b;
    for (size_t i = 0; i < *a.size; ++i) {
        set_a.insert(a.pts[i]);
        set_b.insert(b.pts[i]);
    }
    return set_a == set_b;
}

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

template<> struct fmt::formatter<pmkd::MortonType> {
    formatter<pmkd::MortonType::value_type> int_formatter;

    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return int_formatter.parse(ctx);
    }

    template <typename FormatContext>
    auto format(const pmkd::MortonType& s, FormatContext& ctx) const
    {
        return int_formatter.format(s.code, ctx);
    }
};

void printPMKDInfo(const pmkd::PMKDTree& tree) {
    auto treeInfo = tree.print();
    fmt::print("叶节点数：{}\n", treeInfo.leafNum);
    fmt::print("先序：\n{}\n", treeInfo.preorderTraversal);
    fmt::print("中序：\n{}\n", treeInfo.inorderTraversal);
    fmt::print("指标：\n{}\n", treeInfo.metrics);
    fmt::print("莫顿码：\n");
    for (const auto& morton : treeInfo.leafMortons) {
        std::bitset<sizeof(pmkd::MortonType) * 8> biRepr(morton.code);
        fmt::print("{} ", biRepr.to_string());
    }
    fmt::print("\n");
}

bool savePMKDInfo(const pmkd::PMKDTree& tree, const std::string& filename) {
    auto treeInfo = tree.print(true);

    std::ofstream file(filename);
    if (file.is_open()) {
        // layout
        fmt::print(file, "{}\n", treeInfo.leafNum);
        fmt::print(file, "{}\n", treeInfo.preorderTraversal);
        fmt::print(file, "{}\n", treeInfo.inorderTraversal);
        // interior
        fmt::print(file, "{}\n", treeInfo.splitDim);
        fmt::print(file, "{}\n", treeInfo.splitVal);

        // leaf
        fmt::print(file, "{}\n", treeInfo.metrics);
        fmt::print(file, "{}\n", treeInfo.leafMortons);

        for (const auto& point : treeInfo.leafPoints) {
            fmt::print(file, "({},{},{}) ", point.x, point.y, point.z);
        }
        fmt::print(file, "\n");
        file.close();
        return true;
    }

    fmt::print("无法打开文件\n");
    return false;

}