#include "test_common.h"

using namespace pmkd;

int main(int argc, char* argv[]) {
    int N = argc > 1 ? std::stoi(argv[1]) : 400;
    auto pts = genPts(N, false, false);

    PMKDTree* tree = nullptr;
    auto init = [&]() {
        if (tree) delete tree;
        tree = new PMKDTree();
        tree->firstInsert(pts);
        };
    mTimer("PMKD构造用时", init);
    auto ptsAdd1 = genPts(N / 5, false, false, tree->getGlobalBoundary());
    auto ptsAdd2 = genPts(N / 5, false, false, tree->getGlobalBoundary());

    tree->insert(ptsAdd1);
    tree->insert(ptsAdd2);

    //printPMKD_Plain(*tree);
    //printPMKDInfo(*tree);
    //fmt::print("\n");

    std::ofstream file("correctness.txt");
    if (!file.is_open()) {
        fmt::print("无法打开文件\n");
        return 0;
    }
    // 点查询
    fmt::print("Point Search:\n");

    vector<vec3f> ptQueries;
    ptQueries.append(pts.begin() + pts.size() / 2, pts.end());
    ptQueries.append(ptsAdd1.begin() + ptsAdd1.size() / 2, ptsAdd1.end());
    ptQueries.append(ptsAdd2.begin() + ptsAdd2.size() / 2, ptsAdd2.end());

    int nErr = 0;
    auto ptResp = tree->query(ptQueries);
    for (size_t i = 0; i < ptResp.size(); ++i) {
        if (!ptResp.exist[i]) {
            size_t j = ptResp.queryIdx[i];
            loge("({:.3f}, {:.3f}, {:.3f}) not found", ptQueries[j].x, ptQueries[j].y, ptQueries[j].z);
            ++nErr;
        }
    }
    fmtlog::poll();
    fmt::print("{}/{} Failures\n\n", nErr, ptResp.size());

    // 范围查询
    fmt::print("Range Search:\n");
    fmt::print("Generate Ranges:\n");
    auto rangeQueries = genRanges(tree->primSize() / 3, false, false);
    nErr = 0;
    auto rangeResps = tree->query(rangeQueries);
    // get all inserted points
    vector<vec3f> allPts;
    allPts.append(pts.begin(), pts.end());
    allPts.append(ptsAdd1.begin(), ptsAdd1.end());
    allPts.append(ptsAdd2.begin(), ptsAdd2.end());
    //fmt::print("allPts:\n{}\n", allPts);

    auto rangeResps_Brutal = rangeQuery_Brutal(rangeQueries, allPts);
    for (size_t i = 0; i < rangeResps.size(); ++i) {
        size_t j = rangeResps.queryIdx[i];
        auto rangeResp = rangeResps.at(i);
        auto rangeResp_Brutal = rangeResps_Brutal.at(j);

        bool eq = isContentEqual(rangeResp, rangeResp_Brutal);
        if (!eq) {
            ++nErr;
            loge("Range {} is incorrect", rangeQueries[j].toString());
        }
        // log to file
        if (!eq) fmt::print(file, "Incorrect ");
        fmt::print(file, "Range {}:\n", rangeQueries[j].toString());
        fmt::print(file, "tree: ");
        for (size_t k = 0; k < *(rangeResp.size); ++k) {
            fmt::print(file, "({:.3f}, {:.3f}, {:.3f}) ",
                rangeResp.pts[k].x, rangeResp.pts[k].y, rangeResp.pts[k].z);
        }
        fmt::print(file, "\nbrutal: ");
        for (size_t k = 0; k < *(rangeResp_Brutal.size); ++k) {
            fmt::print(file, "({:.3f}, {:.3f}, {:.3f}) ",
                rangeResp_Brutal.pts[k].x, rangeResp_Brutal.pts[k].y, rangeResp_Brutal.pts[k].z);
        }
        fmt::print(file, "\n\n");
        // end log
    }
    fmtlog::poll();
    fmt::print("{}/{} Failures\n", nErr, rangeQueries.size());

    fmt::print("All done!\n");

    file.close();
    delete tree;

    return 0;
}