#include "test_common.h"

using namespace pmkd;

inline std::string getFilename(const std::string& filename, int turn) {
    return filename.substr(0, filename.size() - 4) + std::to_string(turn) + ".txt";
}

int main(int argc, char* argv[]) {
    int num = argc > 1 ? std::stoi(argv[1]) : 60;
    std::string filename = argc > 2 ? std::string(argv[2]) : "tree.txt";


    auto pts = genPts(num, false, false);

    int turn = 0;
    PMKDTree tree;
    tree.firstInsert(pts);

    bool success = savePMKDInfo(tree, getFilename(filename, turn));
    if (success) fmt::print("成功保存PMKD树{}\n", turn++);

    auto ptsAdd1 = genPts(num / 5, false, false, tree.getGlobalBoundary());
    auto ptsAdd2 = genPts(num / 5, false, false, tree.getGlobalBoundary());

    tree.insert(ptsAdd1);
    success = savePMKDInfo(tree, getFilename(filename, turn));
    if (success) fmt::print("成功保存PMKD树{}\n", turn++);

    tree.insert(ptsAdd2);
    success = savePMKDInfo(tree, getFilename(filename, turn));
    if (success) fmt::print("成功保存PMKD树{}\n", turn++);

    return 0;
}