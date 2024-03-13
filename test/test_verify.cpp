#include <iostream>
#include <string>
#include <auth/verify.h>

#include "test_common.h"

// using K = std::string;
// using V = unsigned long;
// using map_type = parlay::parlay_unordered_map<K, V>;

// int main() {
//     map_type my_map(100,false);
//     my_map.Insert("sue", 1);
//     my_map.Insert("sam", 5);

//     std::cout << "value before increment: " << *my_map.Find("sue") << std::endl;
//     auto increment = [](std::optional<V> v) -> V {return v.has_value() ? 1 + *v : 1;};
//     my_map.Upsert("sue", increment);
//     std::cout << "value after increment: " << *my_map.Find("sue") << std::endl;

//     std::cout << "size before remove: " << my_map.size() << std::endl;
//     my_map.Remove("sue");
//     std::cout << "size after remove: " << my_map.size() << std::endl;

//     std::cout << "size before parallel insertion: " << my_map.size() << std::endl;
//     parlay::parallel_for(0, 20, [&my_map](size_t i) {
//         my_map.Insert(std::to_string(i), i);
//     });
//     std::cout << "size after parallel insertion: " << my_map.size() << std::endl;

//     return 0;
// }

using namespace pmkd;

int main(int argc, char* argv[]) {
    int N = argc > 1 ? std::stoi(argv[1]) : 60;
    bool verbose = argc > 2 ? std::string(argv[2]) == "-v" : false;
#ifdef ENABLE_MERKLE
    AABB bound(-30, -30, -30, 30, 30, 30);
    PMKD_Config config;
    config.globalBoundary = bound;

    auto pts = genPts(N, false, false, bound);

    fmt::print("插入-删除-插入\n");
    PMKDTree* tree = new PMKDTree(config);

    auto remove = [&](auto&& _ptsRemove) {
        tree->remove(_ptsRemove);
    };

    auto ptsAdd1 = genPts(N / 5, false, false, bound);
    auto ptsAdd2 = genPts(N / 5, false, false, bound);

    vector<vec3f> ptRemove(pts.begin(), pts.begin() + pts.size() / 2);

    tree->firstInsert(pts);
    mTimer("删除用时", remove, ptRemove);
    tree->insert(ptsAdd1);
    tree->insert(ptsAdd2);

    // 范围测试
    fmt::print(" Verifiable Range Search:\n");
    auto rangeQueries = genRanges(std::min(1000lu, tree->primSize() / 3), false, false);
    VerifiableRangeQueryResponses veriResps;
    mTimer("平均查询用时", 1.0 / rangeQueries.size(), [&] {
        veriResps = tree->verifiableQuery(rangeQueries);
    });

    int nErr = 0;
    parlay::parlay_unordered_map<int, size_t> table(0.25*N);
    hash_t rootHash = tree->getRootHash();
    mTimer("平均验证用时-单核", 1.0 / veriResps.size(), [&] {
        for (size_t i = 0; i < veriResps.size(); ++i) {
            size_t j = veriResps.queryIdx[i];
            bool correct = verifyRangeQuery_Sequential(rootHash, rangeQueries[j], veriResps, i, table);
            nErr += 1 - correct;
            if (!correct && verbose) {
                fmt::print("Incorrect Range {}:\n", rangeQueries[j].toString());
            }
        }
    });
    fmt::print("{}/{} Failures\n", nErr, veriResps.size());

    nErr = 0;
    mTimer("平均验证用时-并行", 1.0 / veriResps.size(), [&] {
        for (size_t i = 0; i < veriResps.size(); ++i) {
            size_t j = veriResps.queryIdx[i];
            bool correct = verifyRangeQuery(rootHash, rangeQueries[j], veriResps, i, table);
            nErr += 1 - correct;
            if (!correct && verbose) {
                fmt::print("Incorrect Range {}:\n", rangeQueries[j].toString());
            }
        }
    });
    fmt::print("{}/{} Failures\n", nErr, veriResps.size());

    delete tree;
#endif
    
    return 0;
}