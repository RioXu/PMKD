#include "test_common.h"

using namespace pmkd;

int main(int argc, char* argv[]) {
    int num = argc > 1 ? std::stoi(argv[1]) : 10;
    std::string filename = argc > 2 ? std::string(argv[2]) : "tree.txt";

    auto pts = genPts(num, false, false);
    PMKDTree tree;
    tree.insert(std::move(pts));
    bool success = savePMKDInfo(tree, filename);
    if (success) fmt::print("成功保存PMKD树\n");

    return 0;
}