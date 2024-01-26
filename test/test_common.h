#pragma once
#ifndef M_DEBUG
#define M_DEBUG
#endif

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
#include <common/util/utils.h>
#include <tree/kernel.h>
#include <pm_kdtree.h>


using namespace pmkd;

parlay::sequence<vec3f> genPts(size_t n, bool random = true, bool printInfo = true, AABB bound = AABB()) {
    parlay::sequence<vec3f> points;
    points.reserve(n);

    unsigned seed = random ? std::chrono::system_clock::now().time_since_epoch().count() : 0;
    std::mt19937 gen(seed); // 以随机数种子初始化 Mersenne Twister 伪随机数生成器  
    std::normal_distribution<mfloat> dis(0.0, 5.0);

    // 生成n个随机点
    if (bound == AABB()) {
        for (int i = 0; i < n; ++i) {
            vec3f pt(dis(gen), dis(gen), dis(gen));
            points.emplace_back(pt);
        }
    }
    else {
        for (int i = 0; i < n; ++i) {
            vec3f pt(dis(gen), dis(gen), dis(gen));
            while (!bound.include(pt)) {
                pt = vec3f(dis(gen), dis(gen), dis(gen));
            }
            points.emplace_back(pt);
        }
    }

    if (printInfo) {
        for (const auto& pt : points) {
            fmt::printf("(%.4f, %.4f, %.4f) ", pt.x, pt.y, pt.z);
        }
        fmt::printf("\n");
    }
    return points;
}

parlay::sequence<vec3f> sortPts(const parlay::sequence<vec3f>& pts, AABB bound = AABB()) {
    size_t size = pts.size();
    if (pts.empty()) return {};
    if (bound == AABB()) {
        bound = reduce<AABB>(pts, MergeOp());
    }

    auto primIdx = parlay::tabulate(size, [](int i) {return i;});
    vector<MortonType> morton(size);

    // calculate morton code
    parlay::parallel_for(0, size,
        [&](size_t i) { BuildKernel::calcMortonCodes(i, size, pts.data(), &bound, morton.data()); }
    );

    // reorder leaves using morton code
    // note: there are multiple sorting algorithms to choose from
    parlay::integer_sort_inplace(
        primIdx,
        [&](const auto& idx) {return morton[idx].code;});

    auto ptsSorted = parlay::tabulate(size, [&](int i) {return pts[primIdx[i]];});
    return ptsSorted;
}

parlay::sequence<AABB> genRanges(size_t n, bool random = true, bool printInfo = true) {
    parlay::sequence<AABB> ranges;
    ranges.reserve(n);

    unsigned seed = random ? std::chrono::system_clock::now().time_since_epoch().count() : 0;
    std::mt19937 gen(seed); // 以随机数种子初始化 Mersenne Twister 伪随机数生成器  
    std::normal_distribution<mfloat> dis1(0.0, 5.0);
    std::uniform_real_distribution<mfloat> dis2(0.1, 3.0);

    // 生成n个随机AABB
    for (int i = 0; i < n; ++i) {
        vec3f center(dis1(gen), dis1(gen), dis1(gen));
        vec3f lengthVec(dis2(gen), dis2(gen), dis2(gen));
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

RangeQueryResponses rangeQuery_Brutal(
    const parlay::sequence<RangeQuery>& queries, const parlay::sequence<vec3f>& pts) {
    if (queries.empty()) return RangeQueryResponses(0);

    size_t nq = queries.size();
    RangeQueryResponses responses(nq);

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

bool isContentEqual(const RangeQueryResponse& a, const RangeQueryResponse& b) {
    bool isNull_a = !a.pts && !a.size;
    bool isNull_b = !b.pts && !b.size;
    if (isNull_a && isNull_b) return true;
    if ((isNull_a && !isNull_b) || (!isNull_a && isNull_b)) return false;
    // both are not null
    if (*a.size != *b.size) return false;

    std::unordered_set<vec3f, VecHash<vec3f>> set_a;
    std::unordered_set<vec3f, VecHash<vec3f>> set_b;
    for (size_t i = 0; i < *a.size; ++i) {
        set_a.insert(a.pts[i]);
        set_b.insert(b.pts[i]);
    }
    return set_a == set_b;
}


template<> struct fmt::formatter<MortonType> {
    formatter<MortonType::value_type> int_formatter;

    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return int_formatter.parse(ctx);
    }

    template <typename FormatContext>
    auto format(const MortonType& s, FormatContext& ctx) const
    {
        return int_formatter.format(s.code, ctx);
    }
};

void printPMKDInfo(const PMKDTree& tree) {
    auto treeInfo = tree.print();
    fmt::print("叶节点数：{}\n", treeInfo.leafNum);
    fmt::print("先序：\n{}\n", treeInfo.preorderTraversal);
    fmt::print("中序：\n{}\n", treeInfo.inorderTraversal);
    fmt::print("指标：\n{}\n", treeInfo.metrics);
    fmt::print("莫顿码：\n");
    for (const auto& morton : treeInfo.leafMortons) {
        std::bitset<sizeof(MortonType) * 8> biRepr(morton.code);
        fmt::print("{} ", biRepr.to_string());
    }
    fmt::print("\n");
}

bool savePMKDInfo(const PMKDTree& tree, const std::string& filename) {
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