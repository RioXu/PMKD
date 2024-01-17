#include "test_common.h"

using namespace pmkd;

int main() {
    std::vector scales{ 1e4,1e5,1e6,3e6,1e7 };

    PMKDTree* tree = nullptr;
    auto init = [](PMKDTree* tree, auto&& pts) {
        if (tree) delete tree;
        tree = new PMKDTree();
        tree->insert(pts);
    };
    
    for (const auto& scale : scales) {
        auto pts = genPts(scale, true, false);
        fmt::print("规模{}", scale);
        mTimer("构造用时", init, tree, std::move(pts));

        if (tree) delete tree;
    }

    return 0;
}