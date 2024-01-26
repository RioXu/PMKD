#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include <common/util/utils.h>
#include <tree/helper.h>
#include <tree/pm_kdtree.h>
#include <tree/kernel.h>

namespace pmkd {
    void PMKDTree::findBin_Experiment(const vector<vec3f>& ptsAdd, int version) {
        // v1: findBin前不排序，对bin用比较型sort
        // v2：findBin前排序，对bin用radix sort
        // v3：findBin前排序，对bin用比较型sort
        // v4: 在v2基础上，加入findBin找到的叶节点一起排序

        // 结果：v1最慢，v2和v3相仿
        // 占比最大部分为第一次排序primIdx，排序ptsAdd和执行findBin，总计占超80%用时
        size_t sizeInc = ptsAdd.size();
        auto binIdx = bufferPool->acquire<int>(sizeInc);

        vector<vec3f> ptsSorted;
        vector<int> primIdx;
        vector<MortonType> morton;
        vector<MortonType> mortonSorted;

        // calculate morton code
        mTimer("分配与计算Morton", [&] {
            morton.resize(sizeInc);

            parlay::parallel_for(0, sizeInc,
                [&](size_t i) { BuildKernel::calcMortonCodes(i, sizeInc, ptsAdd.data(), &globalBoundary, morton.data()); }
            );});

        const vec3f* targetPts = nullptr;
        if (version == 1) targetPts = ptsAdd.data();
        else if (version == 2 || version == 3 || version == 4) {
            mTimer("分配primIdx", [&] {primIdx = parlay::tabulate(sizeInc, [](int i) {return i;});});
            mTimer("排序primIdx", [&] {
                //note: there are multiple sorting algorithms to choose from
                parlay::integer_sort_inplace(
                    primIdx,
                    [&](const auto& idx) {return morton[idx].code;});
                });
            mTimer("排序ptsAdd", [&] {ptsSorted = parlay::tabulate(sizeInc, [&](int i) {return ptsAdd[primIdx[i]];});});
            if (version == 2 || version == 4) {
                mTimer("排序Morton", [&] {
                    mortonSorted = parlay::tabulate(sizeInc, [&](int i) {return morton[primIdx[i]];});
                });
            }

            targetPts = ptsSorted.data();
        }

        //note: 可用findLeafBin重载 或 reduce获取maxBin
        int maxBin = -1;
        mTimer("执行findBin", [&] {
            parlay::parallel_for(0, sizeInc,
            [&](size_t i) {
                    UpdateKernel::findLeafBin(
                        i, sizeInc, targetPts, primSize(),
                        interiors.getRawRepr(), leaves.getRawRepr(), sceneBoundary, binIdx.data());
                });
        maxBin = parlay::reduce(binIdx, parlay::maximum<int>());
            });


        size_t offset = primSize();
        parlay::parallel_for(0, sizeInc, [&](uint32_t i) {return primIdx[i] = offset + i;});

        if (version == 1) {
            // tested faster than sort_inplace
            parlay::stable_sort_inplace(
                primIdx,
                [&](const auto& idx1, const auto& idx2) {
                    return binIdx[idx1 - offset] < binIdx[idx2 - offset] ||
                        (binIdx[idx1 - offset] == binIdx[idx2 - offset] &&
                            mortonSorted[idx1 - offset].code < mortonSorted[idx2 - offset].code
                        );
                });
        }
        else if (version == 2) {
            mTimer("根据binIdx排序", [&] {
                parlay::stable_integer_sort_inplace(
                    primIdx,
                    [&](const auto& idx) {return static_cast<uint32_t>(binIdx[idx - offset]);});
                });
        }
        else if (version == 3) {
            parlay::stable_sort_inplace(
                primIdx,
                [&](const auto& idx1, const auto& idx2) {return binIdx[idx1 - offset] < binIdx[idx2 - offset];});
        }
        else if (version == 4) {
            MortonType* mortonArr[2] = { leaves.morton.data(), mortonSorted.data() };
            auto getMorton = [&](int idx) {
                return mortonArr[idx / ptNum][idx % ptNum];
            };

            vector<int> binCount;
            vector<int> leafIdx;
            mTimer("统计binCount", [&] {
                binCount = parlay::histogram_by_index(binIdx, maxBin + 1);
                parlay::parallel_for(0, binCount.size(), [&](size_t i) {
                    if (binCount[i] > 0) binCount[i]++;
                });
            });
            // mTimer("统计leafIdx", [&] {
            //     leafIdx.reserve(sizeInc);
            //     for (size_t i = 0; i < binCount.size(); i++) {
            //         if (binCount[i] > 0) {
            //             binCount[i]++;
            //             leafIdx.push_back(i);
            //         }
            //     }
            //     });
            mTimer("统计leafIdx", [&] {
                leafIdx = parlay::remove_duplicate_integers(binIdx, maxBin + 1);
            });

            vector<int> combinedPrimIdx(sizeInc + leafIdx.size());
            vector<int> combinedBinIdx(sizeInc + leafIdx.size());
            mTimer("并行merge", [&] {
                mergeZip(primIdx, leafIdx, binIdx, combinedPrimIdx, combinedBinIdx,
                [&](const auto& idx1, const auto& idx2) {return getMorton(idx1).code < getMorton(idx2).code;});
            });
       
            auto tempIdx = parlay::tabulate(combinedPrimIdx.size(), [&](int i) {return i;});
            mTimer("根据合并binIdx排序", [&] {
                parlay::stable_integer_sort_inplace(
                    tempIdx,
                    [&](const auto& idx) {return static_cast<uint32_t>(combinedBinIdx[idx]);});
            });

            // check correctness, not for performance comparison
            // auto finalBinIdx = parlay::tabulate(combinedPrimIdx.size(), [&](int i) {return combinedBinIdx[tempIdx[i]];});
            // auto finalMorton = parlay::tabulate(combinedPrimIdx.size(), [&](int i) {
            //     int j = combinedPrimIdx[tempIdx[i]];
            //     return getMorton(j).code;
            // });
            // fmt::print("maxBin: {}\n", maxBin);
            // fmt::print("primIdx:\n{}\n", primIdx);
            // fmt::print("leafIdx:\n{}\n", leafIdx);
            // fmt::print("binIdx:\n{}\n", binIdx);
            // fmt::print("combinedPrimIdx:\n{}\n", combinedPrimIdx);
            // fmt::print("combinedBinIdx:\n{}\n", combinedBinIdx);
            // fmt::print("finalBinIdx:\n{}\n", finalBinIdx);
            // fmt::print("finalMorton:\n{}\n", finalMorton);
        }

        bufferPool->release(std::move(binIdx));
    }
}