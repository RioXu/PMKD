#include "test_common.h"

using namespace pmkd;

int main(int argc, char* argv[]) {
    int N = argc > 1 ? std::stoi(argv[1]) : 80;
    bool verbose = argc > 2 ? std::string(argv[2]) == "-v" : false;
    bool writeToFile = argc > 3 ? std::string(argv[3]) == "-w" : false;

    auto pts = genPts(N, false, false);

    fmt::print("插入测试-分批插入\n");
    PMKDTree* tree = nullptr;
    auto init = [&]() {
        if (tree) delete tree;
        tree = new PMKDTree();
        tree->firstInsert(pts);
    };
    auto remove = [&](auto&& _ptsRemove) {
        tree->remove(_ptsRemove);
    };
    
    init();
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
    ptQueries.insert(ptQueries.end(), pts.begin() + pts.size() / 2, pts.end());
    ptQueries.insert(ptQueries.end(), ptsAdd1.begin() + ptsAdd1.size() / 2, ptsAdd1.end());
    ptQueries.insert(ptQueries.end(), ptsAdd2.begin() + ptsAdd2.size() / 2, ptsAdd2.end());

    int nErr = 0;
    auto ptResp = tree->query(ptQueries);
    for (size_t i = 0; i < ptResp.size(); ++i) {
        if (!ptResp.exist[i]) {
            size_t j = ptResp.queryIdx[i];
            if (verbose) loge("({:.3f}, {:.3f}, {:.3f}) not found", ptQueries[j].x, ptQueries[j].y, ptQueries[j].z);
            ++nErr;
        }
    }
    fmtlog::poll();
    fmt::print("{}/{} Failures\n\n", nErr, ptResp.size());

    // 范围查询
    fmt::print("Range Search:\n");
    //fmt::print("Generate Ranges:\n");
    auto rangeQueries = genRanges(tree->primSize() / 3, false, false);
    nErr = 0;
    auto rangeResps = tree->query(rangeQueries);
    // get all inserted points
    vector<vec3f> allPts;
    allPts.insert(allPts.end(), pts.begin(), pts.end());
    allPts.insert(allPts.end(), ptsAdd1.begin(), ptsAdd1.end());
    allPts.insert(allPts.end(), ptsAdd2.begin(), ptsAdd2.end());
    //fmt::print("allPts:\n{}\n", allPts);

    auto rangeResps_Brutal = rangeQuery_Brutal(rangeQueries, allPts);
    for (size_t i = 0; i < rangeResps.size(); ++i) {
        size_t j = rangeResps.queryIdx[i];
        auto rangeResp = rangeResps.at(i);
        auto rangeResp_Brutal = rangeResps_Brutal.at(j);

        bool eq = isContentEqual(rangeResp, rangeResp_Brutal);
        if (!eq) {
            ++nErr;
            if (verbose) loge("Range {} is incorrect", rangeQueries[j].toString());
        }
        // log to file
        if (!eq && writeToFile) {
            fmt::print(file, "Incorrect Range {}:\n", rangeQueries[j].toString());
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
        }
        // end log
    }
    fmtlog::poll();
    fmt::print("{}/{} Failures\n\n", nErr, rangeQueries.size());

    // 删除测试
    vector<vec3f> ptRemove;
    ptRemove.insert(ptRemove.end(), pts.begin() + pts.size() / 2, pts.end());
    ptRemove.insert(ptRemove.end(), ptsAdd1.begin() + ptsAdd1.size() / 2, ptsAdd1.end());
    ptRemove.insert(ptRemove.end(), ptsAdd2.begin() + ptsAdd2.size() / 2, ptsAdd2.end());

    vector<vec3f> ptRemain;
    ptRemain.insert(ptRemain.end(), pts.begin(), pts.begin() + pts.size() / 2);
    ptRemain.insert(ptRemain.end(), ptsAdd1.begin(), ptsAdd1.begin() + ptsAdd1.size() / 2);
    ptRemain.insert(ptRemain.end(), ptsAdd2.begin(), ptsAdd2.begin() + ptsAdd2.size() / 2);

    fmt::print("删除测试-静态树\n");
    // static
    tree->destroy();
    tree->firstInsert(allPts);
    mTimer("删除用时", remove, ptRemove);
    //tree->remove(ptRemove);

    // 点测试
    fmt::print("Point Search:\n");
    nErr = 0;
    ptResp = tree->query(ptRemove);
    for (size_t i = 0; i < ptResp.size(); ++i) {
        if (ptResp.exist[i]) {
            size_t j = ptResp.queryIdx[i];
            if (verbose) loge("({:.3f}, {:.3f}, {:.3f}) removed but found", ptRemove[j].x, ptRemove[j].y, ptRemove[j].z);
            ++nErr;
        }
    }
    ptResp = tree->query(ptRemain);
    for (size_t i = 0; i < ptResp.size(); ++i) {
        if (!ptResp.exist[i]) {
            size_t j = ptResp.queryIdx[i];
            if (verbose) loge("({:.3f}, {:.3f}, {:.3f}) not found", ptRemain[j].x, ptRemain[j].y, ptRemain[j].z);
            ++nErr;
        }
    }
    fmtlog::poll();
    fmt::print("{}/{} Failures\n\n", nErr, ptRemove.size() + ptRemain.size());

    // 范围测试
    fmt::print("Range Search:\n");
    nErr = 0;
    rangeResps = tree->query(rangeQueries);

    rangeResps_Brutal = rangeQuery_Brutal(rangeQueries, ptRemain);

    if (writeToFile) fmt::print(file, "Static Insert + Remove:\n");
    for (size_t i = 0; i < rangeResps.size(); ++i) {
        size_t j = rangeResps.queryIdx[i];
        auto rangeResp = rangeResps.at(i);
        auto rangeResp_Brutal = rangeResps_Brutal.at(j);

        bool eq = isContentEqual(rangeResp, rangeResp_Brutal);
        if (!eq) {
            ++nErr;
            if (verbose) loge("Range {} is incorrect", rangeQueries[j].toString());
        }
        // log to file
        if (!eq && writeToFile) {
            fmt::print(file, "Incorrect Range {}:\n", rangeQueries[j].toString());
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
        }
    }
    fmtlog::poll();
    fmt::print("{}/{} Failures\n\n", nErr, rangeQueries.size());

    fmt::print("删除测试-动态树\n");
    // dynamic
    tree->destroy();

    tree->firstInsert(pts);
    tree->insert(ptsAdd1);
    tree->insert(ptsAdd2);
    mTimer("删除用时", remove, ptRemove);
    //tree->remove(ptRemove);

    // 点测试
    fmt::print("Point Search:\n");
    nErr = 0;
    ptResp = tree->query(ptRemove);
    for (size_t i = 0; i < ptResp.size(); ++i) {
        if (ptResp.exist[i]) {
            size_t j = ptResp.queryIdx[i];
            if (verbose) loge("({:.3f}, {:.3f}, {:.3f}) removed but found", ptRemove[j].x, ptRemove[j].y, ptRemove[j].z);
            ++nErr;
        }
    }
    ptResp = tree->query(ptRemain);
    for (size_t i = 0; i < ptResp.size(); ++i) {
        if (!ptResp.exist[i]) {
            size_t j = ptResp.queryIdx[i];
            if (verbose) loge("i: {}, ({:.3f}, {:.3f}, {:.3f}) not found", i, ptRemain[j].x, ptRemain[j].y, ptRemain[j].z);
            ++nErr;
        }
    }
    fmtlog::poll();
    fmt::print("{}/{} Failures\n\n", nErr, ptRemove.size() + ptRemain.size());

    // 范围测试
    fmt::print("Range Search:\n");
    nErr = 0;
    rangeResps = tree->query(rangeQueries);

    rangeResps_Brutal = rangeQuery_Brutal(rangeQueries, ptRemain);

    if (writeToFile) fmt::print(file, "Dynamic Insert + Remove:\n");
    for (size_t i = 0; i < rangeResps.size(); ++i) {
        size_t j = rangeResps.queryIdx[i];
        auto rangeResp = rangeResps.at(i);
        auto rangeResp_Brutal = rangeResps_Brutal.at(j);

        bool eq = isContentEqual(rangeResp, rangeResp_Brutal);
        if (!eq) {
            ++nErr;
            if (verbose) loge("Range {} is incorrect", rangeQueries[j].toString());
        }
        // log to file
        if (!eq && writeToFile) {
            fmt::print(file, "Incorrect Range {}:\n", rangeQueries[j].toString());
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
        }
    }
    fmtlog::poll();
    fmt::print("{}/{} Failures\n\n", nErr, rangeQueries.size());

    fmt::print("删除测试-插入+删除+插入\n");
    vector<vec3f> ptRemove2(pts.begin(), pts.begin() + pts.size() / 2);
    vector<vec3f> ptRemain2(pts.begin() + pts.size() / 2, pts.end());
    ptRemain2.insert(ptRemain2.end(), ptsAdd1.begin(), ptsAdd1.end());
    ptRemain2.insert(ptRemain2.end(), ptsAdd2.begin(), ptsAdd2.end());

    // dynamic
    tree->destroy();

    tree->firstInsert(pts);
    mTimer("删除用时", remove, ptRemove2);
    tree->insert(ptsAdd1);
    tree->insert(ptsAdd2);

    // 点测试
    fmt::print("Point Search:\n");
    nErr = 0;
    ptResp = tree->query(ptRemove2);
    for (size_t i = 0; i < ptResp.size(); ++i) {
        if (ptResp.exist[i]) {
            size_t j = ptResp.queryIdx[i];
            if (verbose) loge("({:.3f}, {:.3f}, {:.3f}) removed but found", ptRemove2[j].x, ptRemove2[j].y, ptRemove2[j].z);
            ++nErr;
        }
    }
    ptResp = tree->query(ptRemain2);
    for (size_t i = 0; i < ptResp.size(); ++i) {
        if (!ptResp.exist[i]) {
            size_t j = ptResp.queryIdx[i];
            if (verbose) loge("i: {}, ({:.3f}, {:.3f}, {:.3f}) not found", i, ptRemain2[j].x, ptRemain2[j].y, ptRemain2[j].z);
            ++nErr;
        }
    }
    fmtlog::poll();
    fmt::print("{}/{} Failures\n\n", nErr, ptRemove2.size() + ptRemain2.size());

    fmt::print("All done!\n");

    file.close();
    delete tree;

    return 0;
}