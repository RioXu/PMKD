#include "test_common.h"

using namespace pmkd;

int main() {
    std::vector scales{ 1e5,1e6 };
    const int nIter = 20;

    PMKDTree* tree = nullptr;
    auto init = [](PMKDTree* tree, const PMKD_Config& config, auto&& pts) {
        if (tree) delete tree;
        tree = new PMKDTree(config);
        tree->insert(pts);
    };

    fmt::print("非优化版：\n");
    PMKD_Config config;
    config.optimize = false;
    for (const auto& scale : scales) {
        auto pts = genPts(scale, false, false);
        fmt::print("规模{}", scale);
        mTimerRepeated("平均构造用时", nIter, init, tree, config, std::move(pts));

        if (tree) delete tree;
    }

    fmt::print("优化版：\n");
    config.optimize = true;
    for (const auto& scale : scales) {
        auto pts = genPts(scale, false, false);
        fmt::print("规模{}", scale);
        mTimerRepeated("平均构造用时", nIter, init, tree, config, std::move(pts));

        if (tree) delete tree;
    }
    return 0;
}