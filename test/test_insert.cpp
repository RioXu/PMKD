#include "test_common.h"

using namespace pmkd;

int main(int argc, char* argv[]) {
    int N = argc > 1 ? std::stoi(argv[1]) : 30;
    std::string filename = argc > 2 ? std::string(argv[2]) : "tree.txt";

    auto pts = genPts(N, false, false);
    PMKDTree tree;
    tree.firstInsert(std::move(pts));

    auto ptsAdd = genPts(N/10, true, false, tree.getGlobalBoundary());
    tree.insert(ptsAdd);

    return 0;
}