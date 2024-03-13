#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/hash_table.h>

#include <common/util/utils.h>
#include <tree/helper.h>
#include <tree/pm_kdtree.h>
#include <tree/kernel.h>

namespace pmkd {
    void PMKDTree::findBin_Experiment(const vector<vec3f>& ptsAdd, int version, bool print) {
        // v1: findBin前不排序，对bin用比较型sort
        // v2：findBin前排序，对bin用radix sort
        // v3：findBin前排序，对bin用比较型sort
        // v4: 在v2基础上，加入findBin找到的叶节点一起排序

        // 结果：v1最慢，v2和v3相仿
        // 占比最大部分为第一次排序primIdx，排序ptsAdd和执行findBin，总计占超80%用时
        auto& leaves = nodeMgr->getLeaves(0);
        auto& interiors = nodeMgr->getInteriors(0);
        size_t ptNum = nodeMgr->numLeaves();

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
            mTimer("分配primIdx", [&] {
                primIdx.resize(sizeInc);
                parlay::parallel_for(0, sizeInc, [&](size_t i) {primIdx[i] = i;});
            });
            mTimer("排序primIdx", [&] {
                //note: there are multiple sorting algorithms to choose from
                parlay::integer_sort_inplace(
                    primIdx,
                    [&](const auto& idx) {return morton[idx].code;});
                });
            mTimer("排序ptsAdd", [&] {
                ptsSorted.resize(sizeInc);
                parlay::parallel_for(0, sizeInc, [&](size_t i) {ptsSorted[i] = ptsAdd[primIdx[i]];});
            });
            if (version == 2 || version == 4) {
                mTimer("排序Morton", [&] {
                    mortonSorted.resize(sizeInc);
                    parlay::parallel_for(0, sizeInc, [&](size_t i) {mortonSorted[i] = morton[primIdx[i]];});
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
                        interiors.getRawRepr(), leaves.getRawRepr(),
                        binIdx.data());
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
            const MortonType* mortonArr[2] = { leaves.morton.data(), mortonSorted.data() };
            auto getMorton = [&](int idx) {
                return mortonArr[idx / ptNum][idx % ptNum];
            };

            vector<int> binCount;
            parlay::sequence<int> leafIdx;


            mTimer("统计leafIdx", [&] {
                leafIdx = parlay::remove_duplicate_integers(binIdx, maxBin + 1);
            });

            // vector<int> binIdxSorted;
            // mTimer("统计binCount", [&] {
            //     binIdxSorted = parlay::integer_sort(binIdx, [](const auto& idx) { return static_cast<uint32_t>(idx);});
            //     binCount.resize(leafIdx.size());
            //     parlay::parallel_for(0, leafIdx.size(), [&](size_t i) {
            //         const auto& [_l, _r] = std::equal_range(binIdxSorted.begin(), binIdxSorted.end(), leafIdx[i]);
            //         binCount[i] = (_r - _l) + 1;
            //     });
            // });

            // mTimer("统计binCount", [&] {
            //     binCount = parlay::histogram_by_index(binIdx, maxBin + 1);
            //     parlay::parallel_for(0, binCount.size(), [&](size_t i) {
            //         if (binCount[i] > 0) binCount[i]++;
            //     });
            // });

            vector<int> combinedPrimIdx(sizeInc + leafIdx.size());
            vector<int> combinedBinIdx(sizeInc + leafIdx.size());
            mTimer("并行merge", [&] {
                mergeZip(primIdx, leafIdx, binIdx, combinedPrimIdx, combinedBinIdx,
                [&](const auto& idx1, const auto& idx2) {return getMorton(idx1).code < getMorton(idx2).code;});
            });
       
            auto tempIdx = parlay::tabulate(combinedPrimIdx.size(), [&](int i) {return i;});
            parlay::sequence<int> finalBinIdx;
            mTimer("根据合并binIdx排序", [&] {
                parlay::stable_integer_sort_inplace(
                    tempIdx,
                    [&](const auto& idx) {return static_cast<uint32_t>(combinedBinIdx[idx]);});
                
                finalBinIdx = parlay::tabulate(combinedPrimIdx.size(), [&](int i) {return combinedBinIdx[tempIdx[i]];});
            });

            // note: there is a faster top-down parallel algorithm
            // not sure how to apply it to GPU
            mTimer("统计binCount", [&] {
                binCount.resize(leafIdx.size());
                parlay::parallel_for(0, leafIdx.size(), [&](size_t i) {
                    const auto& [_l, _r] = std::equal_range(finalBinIdx.begin() + i, finalBinIdx.end(), leafIdx[i]);
                    binCount[i] = (_r - _l);
                });
            });

            // check correctness, not for performance comparison
            if (print) {
                auto combinedPrimMorton = parlay::tabulate(combinedPrimIdx.size(), [&](int i) {
                    int j = combinedPrimIdx[i];
                    return getMorton(j).code;
                    });
                auto finalMorton = parlay::tabulate(combinedPrimIdx.size(), [&](int i) {
                    int j = combinedPrimIdx[tempIdx[i]];
                    return getMorton(j).code;
                    });
                fmt::print("maxBin: {}\n", maxBin);
                fmt::print("primIdx:\n{}\n", primIdx);
                fmt::print("leafIdx:\n{}\n", leafIdx);
                fmt::print("binIdx:\n{}\n", binIdx);
                fmt::print("binCount:\n{}\n", binCount);
                fmt::print("combinedPrimIdx:\n{}\n", combinedPrimIdx);
                fmt::print("combinedBinIdx:\n{}\n", combinedBinIdx);
                fmt::print("combinedPrimMorton:\n{}\n", combinedPrimMorton);
                fmt::print("finalBinIdx:\n{}\n", finalBinIdx);
                fmt::print("finalMorton:\n{}\n", finalMorton);
            }
        }

        bufferPool->release(std::move(binIdx));
    }

    void PMKDTree::execute(const vector<vec3f>& ptsRemove, const vector<vec3f>& ptsAdd) {
        if (ptsRemove.empty()) {
            insert(ptsAdd);
            return;
        }
        if (ptsAdd.empty()) {
            remove(ptsRemove);
            return;
        }
        isStatic = false;

        // remove-----------------------------------
        size_t nRemove = ptsRemove.size();

        vector<vec3f> ptsRemoveSorted;
        const vec3f* target = ptsRemove.data();

        auto removeBinIdx = bufferPool->acquire<int>(nRemove);

        // sort queries to improve cache friendlyness
        if (config.optimize) {
            ptsRemoveSorted = bufferPool->acquire<vec3f>(nRemove);
            sortPts(ptsRemove, ptsRemoveSorted);

            target = ptsRemoveSorted.data();
        }

        auto nodeMgrDevice = nodeMgr->getDeviceHandle();

        parlay::parallel_for(0, nRemove, [&](size_t i) {
            UpdateKernel::removePoints_step1(i, nRemove, target, nodeMgrDevice, primSize(), removeBinIdx.data());
            }
        );

        if (!ptsRemoveSorted.empty()) bufferPool->release(std::move(ptsRemoveSorted));

        // insert------------------------------------
        size_t ptNum = primSize();
        size_t sizeInc = ptsAdd.size();
        // note: memory allocation can be async
        auto binIdx = bufferPool->acquire<int>(sizeInc);
        auto primIdx = bufferPool->acquire<int>(sizeInc);
        auto morton = bufferPool->acquire<MortonType>(sizeInc);
        auto mortonSorted = bufferPool->acquire<MortonType>(sizeInc);
        auto ptsAddSorted = bufferPool->acquire<vec3f>(sizeInc);

        // get scene boundary
        //sceneBoundary.merge(reduce<AABB>(ptsAdd, MergeOp()));
        //assert(globalBoundary.include(sceneBoundary));

        sortPts(ptsAdd, ptsAddSorted, primIdx, morton);
        // sort morton
        parlay::parallel_for(0, sizeInc,
            [&](size_t i) { mortonSorted[i] = morton[primIdx[i]]; }
        );
        bufferPool->release<MortonType>(std::move(morton));

        //auto nodeMgrDevice = nodeMgr->getDeviceHandle();
        // find leaf bin
        int maxBin = -1;
        parlay::parallel_for(0, sizeInc,
            [&](size_t i) {
                UpdateKernel::findLeafBin(
                    i, sizeInc, ptsAddSorted.data(), primSize(),
                    nodeMgrDevice, binIdx.data());
            });
        maxBin = parlay::reduce(binIdx, parlay::maximum<int>());

        // reset primIdx
        parlay::parallel_for(0, sizeInc, [&](uint32_t i) {return primIdx[i] = ptNum + i;});


        auto getMortonCode = [&](int gi) {
            if (gi >= ptNum) return mortonSorted[gi - ptNum].code;

            int iBatch, _offset;
            transformLeafIdx(gi, nodeMgrDevice.sizesAcc, nodeMgrDevice.numBatches, iBatch, _offset);
            return nodeMgrDevice.leavesBatch[iBatch].morton[_offset].code;
            };

        // get leafIdx
        auto leafIdxLeafSorted = parlay::remove_duplicate_integers(binIdx, maxBin + 1);
        auto leafIdxMortonSorted = parlay::integer_sort(leafIdxLeafSorted, [&](const auto& idx) {return getMortonCode(idx);});
        size_t nInsertBin = leafIdxLeafSorted.size();

        size_t batchLeafSize = sizeInc + nInsertBin;
        auto combinedPrimIdx = bufferPool->acquire<int>(batchLeafSize);
        auto combinedBinIdx = bufferPool->acquire<int>(batchLeafSize);
        // parallel merge
        mergeZip(primIdx, leafIdxMortonSorted, binIdx, combinedPrimIdx, combinedBinIdx,
            [&](const auto& idx1, const auto& idx2) {
                return getMortonCode(idx1) < getMortonCode(idx2);
            });


        // allocate memory for final insertion
        // note: can be async
        Leaves leaves;
        leaves.resizePartial(batchLeafSize);
        leaves.treeLocalRangeR.resize(batchLeafSize);
        leaves.derivedFrom.resize(batchLeafSize);


        Interiors interiors;
        interiors.resize(batchLeafSize - 1);

        auto mapidx = bufferPool->acquire<int>(batchLeafSize - 1);
        auto metrics = bufferPool->acquire<uint8_t>(batchLeafSize - 1);
        auto splitDim = bufferPool->acquire<int>(batchLeafSize - 1);
        auto splitVal = bufferPool->acquire<mfloat>(batchLeafSize - 1);
        auto parent = bufferPool->acquire<int>(batchLeafSize - 1);

        // build aid
        auto visitCount = std::vector<AtomicCount>(batchLeafSize - 1);
        auto innerBuf = bufferPool->acquire<int>(batchLeafSize - 1, 0);
        auto leafBuf = bufferPool->acquire<int>(batchLeafSize, 0);

        auto ptsAddFinal = bufferPool->acquire<vec3f>(batchLeafSize);
        auto treeLocalRangeL = bufferPool->acquire<int>(batchLeafSize);
        auto interiorCount = bufferPool->acquire<int>(nInsertBin);
        auto interiorToLeafIdx = bufferPool->acquire<int>(batchLeafSize - nInsertBin);

        auto& tempIdx = primIdx;
        tempIdx.resize(batchLeafSize);
        auto& finalPrimIdx = binIdx;
        finalPrimIdx.resize(batchLeafSize);

        parlay::parallel_for(0, batchLeafSize,
            [&](size_t i) { tempIdx[i] = i; }
        );
        // sort by combinedBinIdx
        parlay::stable_integer_sort_inplace(
            tempIdx,
            [&](const auto& idx) {return static_cast<uint32_t>(combinedBinIdx[idx]);});

        parlay::parallel_for(0, batchLeafSize,
            [&](size_t i) { finalPrimIdx[i] = combinedPrimIdx[tempIdx[i]]; }
        );
        bufferPool->release<int>(std::move(combinedPrimIdx));

        // set derivedFrom
        auto& finalBinIdx = leaves.derivedFrom;
        parlay::parallel_for(0, batchLeafSize,
            [&](size_t i) { finalBinIdx[i] = combinedBinIdx[tempIdx[i]]; }
        );
        bufferPool->release<int>(std::move(tempIdx));
        bufferPool->release<int>(std::move(combinedBinIdx));

        // set final points to add
        parlay::parallel_for(0, batchLeafSize,
            [&](size_t i) {
                int gi = finalPrimIdx[i];
                if (gi >= ptNum) ptsAddFinal[i] = ptsAddSorted[gi - ptNum];
                else {
                    int iBatch, _offset;
                    transformLeafIdx(gi, nodeMgrDevice.sizesAcc, nodeMgrDevice.numBatches, iBatch, _offset);
                    ptsAddFinal[i] = nodeMgrDevice.ptsBatch[iBatch][_offset];
                }
            }
        );
        // set final mortons to add, set removal
        parlay::parallel_for(0, batchLeafSize,
            [&](size_t i) {
                int gi = finalPrimIdx[i];
                if (gi >= ptNum) leaves.morton[i] = mortonSorted[gi - ptNum];
                else {
                    int iBatch, _offset;
                    transformLeafIdx(gi, nodeMgrDevice.sizesAcc, nodeMgrDevice.numBatches, iBatch, _offset);
                    leaves.morton[i] = nodeMgrDevice.leavesBatch[iBatch].morton[_offset];
                    // set removal
                    int r = nodeMgrDevice.leavesBatch[iBatch].replacedBy[_offset];
                    leaves.replacedBy[i] = r;
                }
            }
        );

        bufferPool->release<vec3f>(std::move(ptsAddSorted));
        // note: leaf hash calculation can be boost
        // by recording leafBinIdx2finalPrimIdx and copying hash
        // to avoid some costly hash calculation
        bufferPool->release<int>(std::move(finalPrimIdx));
        bufferPool->release<MortonType>(std::move(mortonSorted));

        // set bin count
        parlay::parallel_for(0, nInsertBin, [&](size_t i) {
            const auto& [_l, _r] = std::equal_range(finalBinIdx.begin(), finalBinIdx.end(), leafIdxLeafSorted[i]);
            interiorCount[i] = _r - _l - 1;
            });
        parlay::scan_inplace(interiorCount);


        // set tree local range
        parlay::parallel_for(0, batchLeafSize, [&](size_t i) {
            int j = std::lower_bound(leafIdxLeafSorted.begin(), leafIdxLeafSorted.end(), finalBinIdx[i]) - leafIdxLeafSorted.begin();
            treeLocalRangeL[i] = interiorCount[j] + j;
            leaves.treeLocalRangeR[i] = j < nInsertBin - 1 ? interiorCount[j + 1] + j + 1 : batchLeafSize;
            });

        // calc metrics
        parlay::parallel_for(0, interiorToLeafIdx.size(), [&](size_t i) {
            int j = std::upper_bound(interiorCount.begin(), interiorCount.end(), i) - interiorCount.begin() - 1;
            interiorToLeafIdx[i] = i + j;
            });

        parlay::parallel_for(0, interiorToLeafIdx.size(), [&](size_t i) {
            DynamicBuildKernel::calcBuildMetrics(i, interiorToLeafIdx.size(), globalBoundary,
            leaves.morton.data(), interiorToLeafIdx.data(),
            metrics.data(), interiors.splitDim.data(), interiors.splitVal.data());
            });

        //---------------------------------------------------------------------
        // build interiors

        BuildAid aid{ metrics.data(), visitCount.data(), innerBuf.data(),leafBuf.data() };

        parlay::parallel_for(0, batchLeafSize,
            [&](size_t i) {
                DynamicBuildKernel::buildInteriors(
                    i, batchLeafSize, treeLocalRangeL.data(), leaves.getRawRepr(), interiors.getRawRepr(), aid);
            }
        );

        bufferPool->release<int>(std::move(treeLocalRangeL));
        bufferPool->release<uint8_t>(std::move(metrics));

        // calculate new indices for interiors
        auto& segLen = leafBuf;
        leaves.segOffset = parlay::scan(segLen).first;

        parlay::parallel_for(0, interiorCount.size() - 1, [&](size_t i) {
            DynamicBuildKernel::interiorMapIdxInit(i, interiorCount.size(), batchLeafSize, interiorCount.data(), mapidx.data());
            });


        parlay::parallel_for(0, interiorToLeafIdx.size(),
            [&](size_t i) {
                DynamicBuildKernel::calcInteriorNewIdx(
                    i, interiorToLeafIdx.size(), interiorToLeafIdx.data(),
                    leaves.getRawRepr(), interiors.getRawRepr(), aid.segLen, aid.leftLeafCount, mapidx.data());
            }
        );
        bufferPool->release<int>(std::move(interiorToLeafIdx));

        // reorder interiors
        auto& rangeL = innerBuf;
        auto& rangeR = leafBuf;
        rangeR.resize(batchLeafSize - 1);

        parlay::parallel_for(0, batchLeafSize - 1,
            [&](size_t i) {
                DynamicBuildKernel::reorderInteriors(
                    i, batchLeafSize - 1, mapidx.data(), interiors.getRawRepr(),
                    rangeL.data(), rangeR.data(),
                    splitDim.data(), splitVal.data(), parent.data());
            }
        );
        bufferPool->release<int>(std::move(interiors.rangeL));
        bufferPool->release<int>(std::move(interiors.rangeR));
        bufferPool->release<int>(std::move(interiors.splitDim));
        bufferPool->release<mfloat>(std::move(interiors.splitVal));
        bufferPool->release<int>(std::move(interiors.parent));

        interiors.rangeL = std::move(rangeL);
        interiors.rangeR = std::move(rangeR);
        interiors.splitDim = std::move(splitDim);
        interiors.splitVal = std::move(splitVal);
        interiors.parent = std::move(parent);

        parlay::parallel_for(0, batchLeafSize,
            [&](size_t i) {
                DynamicBuildKernel::remapLeafParents(
                    i, batchLeafSize, mapidx.data(), leaves.getRawRepr());
            }
        );

        bufferPool->release<int>(std::move(mapidx));

        // set replacedBy of leafbins and parent of subtrees' roots
        parlay::parallel_for(0, nInsertBin, [&](size_t i) {
            int gi = leafIdxLeafSorted[i];
            int iBatch, _offset;
            transformLeafIdx(gi, nodeMgrDevice.sizesAcc, nodeMgrDevice.numBatches, iBatch, _offset);
            auto& binLeaves = nodeMgrDevice.leavesBatch[iBatch];
            const auto& binInteriors = nodeMgrDevice.interiorsBatch[iBatch];

            binLeaves.replacedBy[_offset] = ptNum + interiorCount[i] + i;

            int parentCode = ((gi - _offset) << 1) + binLeaves.parent[_offset];
            interiors.parent[interiorCount[i]] = -parentCode;
            });

#ifdef ENABLE_MERKLE
        // calculate node hash
        parlay::parallel_for(0, batchLeafSize,
            [&](size_t i) {
                BuildKernel::calcLeafHash(i, batchLeafSize, ptsAddFinal.data(), leaves.replacedBy.data(), leaves.hash.data());
            }
        );
        parlay::parallel_for(0, batchLeafSize,
            [&](size_t i) {
                DynamicBuildKernel::calcInteriorHash_Batch(i, batchLeafSize,
                leaves.getRawRepr(), interiors.getRawRepr());
            }
        );
#endif
        // revert removal of bins to insert
        parlay::parallel_for(0, nInsertBin,
            [&](size_t i) {
                UpdateKernel::revertRemoval(i, nInsertBin, leafIdxLeafSorted.data(), nodeMgrDevice);
            }
        );

        // mixed hash update----------------------------------------
        parlay::sequence<int> removeBinExcludeInsert;
        mTimer("忽略混合Bin去重耗时", [&] {
            parlay::hashtable<parlay::hash_numeric<int>> table(nInsertBin, parlay::hash_numeric<int>());
            parlay::parallel_for(0, nInsertBin, [&](size_t i) {
                table.insert(leafIdxLeafSorted[i]);
            });
            removeBinExcludeInsert = parlay::filter(removeBinIdx, [&](int e) {
                return table.find(e) == -1;  // remove bin that is not one of insert bins
            });
        });
        bufferPool->release(std::move(removeBinIdx));

        size_t nr = removeBinExcludeInsert.size();

        //nodeMgrDevice = nodeMgr->getDeviceHandle();
#ifdef ENABLE_MERKLE
        parlay::parallel_for(0, nr, [&](size_t i) {
            UpdateKernel::calcSelectedLeafHash(i, nr, removeBinExcludeInsert.data(), nodeMgrDevice);
        });
#endif
        parlay::parallel_for(0, nr, [&](size_t i) {
            UpdateKernel::removePoints_step2(i, nr, removeBinExcludeInsert.data(), nodeMgrDevice);
        });
#ifdef ENABLE_MERKLE
        // calculate node hash
        parlay::parallel_for(0, interiorCount.size(),
            [&](size_t i) {
                DynamicBuildKernel::calcInteriorHash_Upper(i, interiorCount.size(), interiorCount.data(), leafIdxLeafSorted.data(),
                interiors.getRawRepr(), nodeMgrDevice);
            }
        );
#endif
        bufferPool->release<int>(std::move(interiorCount));

        nodeMgr->append(std::move(leaves), std::move(interiors), std::move(ptsAddFinal));
    }
}