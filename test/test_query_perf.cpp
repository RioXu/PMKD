#include "test_common.h"
#include <tree/kernel.h>

using namespace pmkd;

int main() {
    const size_t N = 1e6;
    auto pts = genPts(N, true, false);
    PMKDTree tree;
    tree.insert(pts);

    std::vector<QueryResponse> ptResp;

    auto query = [&](const PMKDTree& tree, auto&& pts) {
        ptResp = tree.query(pts);
    };

    // 点查询
    // v1: 不排序
    mTimer("Point Search v1 (no sort)", query, tree, pts);
    int nErr = 0;
    for (size_t i = 0; i < ptResp.size(); ++i) {
        if (!ptResp[i].exist) {
            //loge("({:.3f}, {:.3f}, {:.3f}) not found", pts[i].x, pts[i].y, pts[i].z);
            ++nErr;
        }
    }
    //fmtlog::poll();
    fmt::print("{}/{} Failures\n\n", nErr, ptResp.size());

    // v2: 排序
    // prepare
    auto gBoundary = tree.getGlobalBoundary();
    vector<MortonType> mortons;
    vector<int> primIdx;
    vector<vec3f> ptsSorted;

    // do sorting
    auto doSorting = [&] {
        mortons.resize(pts.size());
        primIdx = parlay::to_sequence(parlay::iota<int>(pts.size()));
        parlay::parallel_for(0, pts.size(),
            [&](size_t i) { BuildKernel::calcMortonCodes(i, pts.size(), pts.data(), &gBoundary, mortons.data()); }
        );

        parlay::integer_sort_inplace(primIdx, [&](const auto& idx) {return mortons[idx].code;});
        ptsSorted = parlay::tabulate(pts.size(), [&](int i) {return pts[primIdx[i]];});
    };
    
    mTimer("Sorting", doSorting);
    mTimer("Point Search v2 (sorted)", query, tree, ptsSorted);
    nErr = 0;
    for (size_t i = 0; i < ptResp.size(); ++i) {
        if (!ptResp[i].exist) {
            //loge("({:.3f}, {:.3f}, {:.3f}) not found", pts[i].x, pts[i].y, pts[i].z);
            ++nErr;
        }
    }
    //fmtlog::poll();
    fmt::print("{}/{} Failures\n\n", nErr, ptResp.size());

    // v3: 排序+morton数组查询
    // prepare
    mTimer("Point Search v3 (binary search)",
        [&] {
            auto mortonSorted = parlay::tabulate(pts.size(), [&](int i) {return mortons[primIdx[i]];});
            parlay::parallel_for(0, pts.size(),
                [&](size_t i) {
                    auto it1 = std::lower_bound(
                        tree.leaves.morton.begin(), tree.leaves.morton.end(), mortonSorted[i],
                        [](const auto& e1, const auto& e2) {return e1.code < e2.code;});
                    auto it2 = std::upper_bound(
                        tree.leaves.morton.begin(), tree.leaves.morton.end(), mortonSorted[i],
                        [](const auto& e1, const auto& e2) {return e1.code < e2.code;});
                    size_t _i1 = (it1 - tree.leaves.morton.begin()) / sizeof(MortonType);
                    size_t _i2 = (it2 - tree.leaves.morton.begin()) / sizeof(MortonType);
                    assert(it1 != tree.leaves.morton.end() && it2 != tree.leaves.morton.end());
                    if (it1 == tree.leaves.morton.end() || it2 == tree.leaves.morton.end())
                        ptResp[i].exist = false;
                    else {
                        ptResp[i].exist = false;
                        for (size_t j = _i1;j <= _i2;j++) {
                            if (tree.pts[j] == ptsSorted[i]) {
                                ptResp[i].exist = true;
                                break;
                            }
                        }
                    }
                }
            );
        }
    );
    nErr = 0;
    for (size_t i = 0; i < ptResp.size(); ++i) {
        if (!ptResp[i].exist) {
            //loge("({:.3f}, {:.3f}, {:.3f}) not found", pts[i].x, pts[i].y, pts[i].z);
            ++nErr;
        }
    }
    //fmtlog::poll();
    fmt::print("{}/{} Failures\n\n", nErr, ptResp.size());

    return 0;
}