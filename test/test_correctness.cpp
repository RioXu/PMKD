#include "test_common.h"

using namespace pmkd;

int main() {
    const size_t N = 30;
    auto pts = genPts(N, false, true);
    PMKDTree* tree = nullptr;
    auto init = [&]() {
        if (tree) delete tree;
        tree = new PMKDTree();
        tree->insert(pts);
    };
    mTimer("PMKD构造用时", init);
    printPMKDInfo(*tree);
    fmt::print("\n");

    std::ofstream file("correctness.txt");
    if (!file.is_open()) {
        fmt::print("无法打开文件\n");
        return 0;
    }
    // 点查询
    fmt::print("Point Search:\n");
    int nErr = 0;
    auto ptResp = tree->query(pts);
    for (size_t i = 0; i < ptResp.size(); ++i) {
        if (!ptResp[i].exist) {
            loge("({:.3f}, {:.3f}, {:.3f}) not found", pts[i].x, pts[i].y, pts[i].z);
            ++nErr;
        }
    }
    fmtlog::poll();
    fmt::print("{}/{} Failures\n\n", nErr, ptResp.size());

    // 范围查询
    fmt::print("Range Search:\n");
    fmt::print("Generate Ranges:\n");
    auto rangeQueries = genRanges(N, false, true);
    nErr = 0;
    auto rangeResps = tree->query(rangeQueries);
    auto rangeResps_Brutal = rangeQuery_Brutal(rangeQueries, pts);
    for (size_t i = 0; i < rangeResps.size(); ++i) {
        auto rangeResp = rangeResps.at(i);
        auto rangeResp_Brutal = rangeResps_Brutal.at(i);

        bool eq = isContentEqual(rangeResp, rangeResp_Brutal);
        if (!eq) {
            ++nErr;
            loge("Range {} is incorrect", rangeQueries[i].toString());
        }
        // log to file
        if (!eq) fmt::print(file, "Incorrect ");
        fmt::print(file, "Range {}:\n", rangeQueries[i].toString());
        fmt::print(file, "tree: ");
        for (size_t j = 0; j < *(rangeResp.size); ++j) {
            fmt::print(file, "({:.3f}, {:.3f}, {:.3f}) ",
                rangeResp.pts[j].x, rangeResp.pts[j].y, rangeResp.pts[j].z);
        }
        fmt::print(file, "\nbrutal: ");
        for (size_t j = 0; j < *(rangeResp_Brutal.size); ++j) {
            fmt::print(file, "({:.3f}, {:.3f}, {:.3f}) ",
                rangeResp_Brutal.pts[j].x, rangeResp_Brutal.pts[j].y, rangeResp_Brutal.pts[j].z);
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