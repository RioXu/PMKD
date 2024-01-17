#include "test_common.h"


using namespace pmkd;

/* 单核merge对比并行merge.
   结论：O0优化下并行版显著更快，O3优化下性能相仿
*/
void testParallelReduce() {
    const size_t n = 100000;
    auto pts = genPts(n);
    // 单线程顺序merge计时
    auto start = std::chrono::high_resolution_clock::now();

    AABB box;
    for (size_t i = 0; i < n; ++i) { box.merge(pts[i]); }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;
    std::cout << "单线程顺序merge用时: " << elapsed_time.count() * 1000 << " ms" << std::endl;
    std::cout << "Result: " << box.toString() << std::endl;

    // 多线程并行merge计时
    start = std::chrono::high_resolution_clock::now();

    auto box2 = reduce<AABB>(pts, MergeOp());

    end = std::chrono::high_resolution_clock::now();
    elapsed_time = end - start;
    std::cout << "多线程并行merge用时: " << elapsed_time.count() * 1000 << " ms" << std::endl;
    std::cout << "Result: " << box2.toString() << std::endl;
}


int main() {
    testParallelReduce();
    return 0;
}