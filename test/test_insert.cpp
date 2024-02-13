#include "test_common.h"

using namespace pmkd;

vector<vector<vec3f>> divideInto(const vector<vec3f>& points, float factor=0.25, int nBatches=3) {
    vector<vector<vec3f>> batches(nBatches);
    size_t accSize = (1.0 - factor * (nBatches - 1)) * points.size();
    batches[0].assign(points.begin(), points.begin() + accSize);

    for (int i = 1; i < nBatches; i++) {
        size_t batchSize = factor * points.size();
        size_t end = std::min(points.size(), accSize + batchSize);
        batches[i].assign(points.begin() + accSize, points.begin() + end);
        accSize += batchSize;
    }
    return batches;
}

int main(int argc, char* argv[]) {
    // get a float from arguments
    float addFactor = argc < 2 ? 0.25 : std::stof(argv[1]);
    int nBatches = argc < 3 ? 3 : std::stoi(argv[2]);

    // test scales
    std::vector scales{ 1e4,1e5,1e6,3e6,1e7 };

    AABB bound(-30, -30, -30, 30, 30, 30);
    PMKD_Config config;
    config.globalBoundary = bound;

    PMKDTree* tree = new PMKDTree(config);
    auto staticBuild = [](PMKDTree* tree, auto&& pts) {
        tree->firstInsert(pts);
    };

    auto addFunc = [](PMKDTree* tree, auto&& pts) {
        tree->insert(pts);
    };

    // divide into 3 batches
    auto incrementalBuild = [](PMKDTree* tree, auto&& ptsBatches) {
        tree->firstInsert(ptsBatches[0]);
        for (int i = 1; i < ptsBatches.size(); i++) {
            tree->insert(ptsBatches[i]);
        }
    };

    auto incrementalBuild_v2 = [](PMKDTree* tree, auto&& ptsBatches) {
        tree->firstInsert(ptsBatches[0]);
        for (int i = 1; i < ptsBatches.size(); i++) {
            tree->insert_v2(ptsBatches[i]);
        }
    };

    for (const auto& scale : scales) {
        auto pts = genPts(scale, true, false, bound);
        auto ptsAdd = genPts(scale * addFactor, true, false, bound);
        auto ptsBatches = divideInto(pts, addFactor, nBatches);
        fmt::print("规模{}", scale);
        mTimer("一次构造用时", staticBuild, tree, pts);
        mTimer("插入用时", addFunc, tree, ptsAdd);
        tree->destroy();
        mTimer("分批构造用时", incrementalBuild, tree, ptsBatches);
        mTimer("插入用时", addFunc, tree, ptsAdd);
        tree->destroy();
        mTimer("分批v2构造用时", incrementalBuild_v2, tree, ptsBatches);
        tree->destroy();
    }

    return 0;
}