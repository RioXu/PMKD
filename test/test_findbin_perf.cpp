#include "test_common.h"

using namespace pmkd;

int main(int argc, char* argv[]) {
    int N = argc > 1 ? std::stoi(argv[1]) : 1e7;
    std::string filename = argc > 2 ? std::string(argv[2]) : "tree.txt";

    auto pts = genPts(N, false, false);
    PMKDTree tree;
    tree.firstInsert(std::move(pts));
    //bool success = savePMKDInfo(tree, filename);
    //if (success) fmt::print("成功保存PMKD树\n");


    auto findBin = [&](auto&& ptsAdd, int version) {
        tree.findBin_Experiment(ptsAdd, version);
        };
    // v1
    //auto ptsAdd = genPts(N / 10, true, false, tree.getGlobalBoundary());
    //mTimer("findBin前不排序，对bin用比较型sort", findBin, ptsAdd, 1);
    // v2
    auto ptsAdd = genPts(N / 10, true, false, tree.getGlobalBoundary());
    fmt::print("Tree size: {}\nAdd size: {}\n", tree.primSize(), ptsAdd.size());
    mTimer("findBin前排序，对bin用radix sort", findBin, ptsAdd, 2);
    // v3
    ptsAdd = genPts(N / 10, true, false, tree.getGlobalBoundary());
    //mTimer("findBin前排序，对bin用比较型sort", findBin, ptsAdd, 3);
    mTimer("v4", findBin, ptsAdd, 4);

    // std::ofstream ifile("insert.txt");
    // if (ifile.is_open()) {
    //     auto ptsAddSorted = sortPts(ptsAdd, tree.getGlobalBoundary());
    //     for (const auto& point : ptsAddSorted) {
    //         fmt::print(ifile, "({},{},{}) ", point.x, point.y, point.z);
    //     }
    //     ifile.close();
    // }

    return 0;
}