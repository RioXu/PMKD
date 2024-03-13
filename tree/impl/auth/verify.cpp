#include <fmt/ranges.h>
#include <auth/verify.h>

namespace pmkd {
    struct ChildHash {
        const hash_t* ref[2];
        // index operator
        const hash_t*& operator[](size_t i) { return ref[i]; }

        ChildHash():ref{nullptr, nullptr} {}
    };

    bool verifyRangeQuery(const hash_t& rootHash, const RangeQuery& query, const VerifiableRangeQueryResponses& resps, size_t idx,
        parlay::parlay_unordered_map<int, size_t>& table) {

        using K = int;
        using V = size_t;

        size_t mStart = resps.mOffset[idx], mEnd = idx < resps.size() - 1 ? resps.mOffset[idx + 1] : resps.vs.mNodes.size();
        size_t hStart = resps.hOffset[idx], hEnd = idx < resps.size() - 1 ? resps.hOffset[idx + 1] : resps.vs.hNodes.size();
        size_t fStart = resps.fOffset[idx], fEnd = idx < resps.size() - 1 ? resps.fOffset[idx + 1] : resps.vs.fNodes.size();
        parlay::sequence<AtomicCount> visitCount(mEnd - mStart);
        parlay::sequence<hash_t> mHash(mEnd - mStart);
        parlay::sequence<ChildHash> childHash(mEnd - mStart);
        parlay::sequence<hash_t> lHash(fEnd - fStart);

        // fmt::print("idx: {}\n", idx);
        // if (idx == 13 || idx == 14) {
        //     fmt::print("mKeys:\n{}\n", vector<int>(resps.vs.mNodes.key.begin() + mStart, resps.vs.mNodes.key.begin() + mEnd));
        //     auto printParent = [](const int* arr, size_t start, size_t end) {
        //         fmt::print("[");
        //         for (size_t _i = start; _i < end; _i++) {
        //             int parentCode = arr[_i];
        //             if (parentCode == -1) fmt::print("{}", parentCode);
        //             else {
        //                 int parent = parentCode >> 1;
        //                 uint8_t isRC = parentCode & 1;
        //                 fmt::print("{}:{}", parent, isRC);
        //             }
        //             if (_i != end - 1) fmt::print(", ");
        //         }
        //         fmt::print("]\n");
        //     };
        //     fmt::print("mParents: [{},{})\n", mStart, mEnd);
        //     printParent(resps.vs.mNodes.parentCode.data(), mStart, mEnd);
        //     fmt::print("hParents: [{},{})\n", hStart, hEnd);
        //     printParent(resps.vs.hNodes.parentCode.data(), hStart, hEnd);
        //     fmt::print("fParents: [{},{})\n", fStart, fEnd);
        //     printParent(resps.vs.fNodes.parentCode.data(), fStart, fEnd);
        // }
        parlay::parallel_for(mStart, mEnd, [&](size_t i) {
            table.Insert(resps.vs.mNodes.key[i], i);
        });

        int rootIndex = -1;
        // process H Nodes
        //for (size_t i=hStart; i < hEnd; i++) 
        parlay::parallel_for(hStart, hEnd, [&](size_t i)
        {
            const auto& mNodes = resps.vs.mNodes;

            K key;
            bool isRC;
            decodeParent(resps.vs.hNodes.parentCode[i], key, isRC);
            auto v = table.Find(key);
            //assert(v.has_value());
            size_t midx = *v - mStart;
            childHash[midx][isRC] = &resps.vs.hNodes.hash[i];

            while (visitCount[midx].cnt++ == 1) {
                int k = idx;
                //assert(childHash[midx][0] != nullptr);
                //assert(childHash[midx][1] != nullptr);
                computeDigest(&mHash[midx], childHash[midx][0], childHash[midx][1],
                    mNodes.splitDim[midx + mStart],
                    mNodes.splitVal[midx + mStart],
                    mNodes.removal[midx + mStart]);

                int parentCode = mNodes.parentCode[midx + mStart];
                if (parentCode == -1) {
                    rootIndex = midx;
                    break;
                }
                decodeParent(mNodes.parentCode[midx + mStart], key, isRC);
                v = table.Find(key);
                //assert(v.has_value());
                size_t next = *v - mStart;
                childHash[next][isRC] = &mHash[midx];
                midx = next;
            }
        }
        );
        // bottom up from F Nodes
        parlay::parallel_for(fStart, fEnd, [&](size_t i) {
            const auto& fNodes = resps.vs.fNodes;
            const auto& mNodes = resps.vs.mNodes;
            computeDigest(&lHash[i - fStart], fNodes.pt[i].x, fNodes.pt[i].y, fNodes.pt[i].z, fNodes.removal[i]);

            K key;
            bool isRC;
            decodeParent(fNodes.parentCode[i], key, isRC);

            auto v = table.Find(key);
            //assert(v.has_value());
            size_t midx = *v - mStart;
            childHash[midx][isRC] = &lHash[i - fStart];

            while (visitCount[midx].cnt++ == 1) {
                int k = idx;
                //assert(childHash[midx][0] != nullptr);
                //assert(childHash[midx][1] != nullptr);
                computeDigest(&mHash[midx], childHash[midx][0], childHash[midx][1],
                    mNodes.splitDim[midx + mStart],
                    mNodes.splitVal[midx + mStart],
                    mNodes.removal[midx + mStart]);

                int parentCode = mNodes.parentCode[midx + mStart];
                if (parentCode == -1) {
                    rootIndex = midx;
                    break;
                }
                decodeParent(mNodes.parentCode[midx + mStart], key, isRC);
                v = table.Find(key);
                //assert(v.has_value());
                size_t next = *v - mStart;
                childHash[next][isRC] = &mHash[midx];
                midx = next;
            }
        });
        // clean up
        table.clear();

        return rootIndex < 0 || equal(mHash[rootIndex], rootHash);
    }

    bool verifyRangeQuery_Sequential(const hash_t& rootHash, const RangeQuery& query, const VerifiableRangeQueryResponses& resps, size_t idx,
        parlay::parlay_unordered_map<int, size_t>& table) {

        using K = int;
        using V = size_t;

        size_t mStart = resps.mOffset[idx], mEnd = idx < resps.size() - 1 ? resps.mOffset[idx + 1] : resps.vs.mNodes.size();
        size_t hStart = resps.hOffset[idx], hEnd = idx < resps.size() - 1 ? resps.hOffset[idx + 1] : resps.vs.hNodes.size();
        size_t fStart = resps.fOffset[idx], fEnd = idx < resps.size() - 1 ? resps.fOffset[idx + 1] : resps.vs.fNodes.size();
        parlay::sequence<AtomicCount> visitCount(mEnd - mStart);
        parlay::sequence<hash_t> mHash(mEnd - mStart);
        parlay::sequence<ChildHash> childHash(mEnd - mStart);
        parlay::sequence<hash_t> lHash(fEnd - fStart);


        for (size_t i = mStart; i < mEnd; i++) {
            table.Insert(resps.vs.mNodes.key[i], i);
        }

        int rootIndex = -1;
        // process H Nodes
        for (size_t i = hStart; i < hEnd; i++) {
            const auto& mNodes = resps.vs.mNodes;

            K key;
            bool isRC;
            decodeParent(resps.vs.hNodes.parentCode[i], key, isRC);
            auto v = table.Find(key);
            //assert(v.has_value());
            size_t midx = *v - mStart;
            childHash[midx][isRC] = &resps.vs.hNodes.hash[i];

            while (visitCount[midx].cnt++ == 1) {
                int k = idx;
                //assert(childHash[midx][0] != nullptr);
                //assert(childHash[midx][1] != nullptr);
                computeDigest(&mHash[midx], childHash[midx][0], childHash[midx][1],
                    mNodes.splitDim[midx + mStart],
                    mNodes.splitVal[midx + mStart],
                    mNodes.removal[midx + mStart]);

                int parentCode = mNodes.parentCode[midx + mStart];
                if (parentCode == -1) {
                    rootIndex = midx;
                    break;
                }
                decodeParent(mNodes.parentCode[midx + mStart], key, isRC);
                v = table.Find(key);
                //assert(v.has_value());
                size_t next = *v - mStart;
                childHash[next][isRC] = &mHash[midx];
                midx = next;
            }
        }

        // bottom up from F Nodes
        for (size_t i = fStart; i < fEnd; i++) {
            const auto& fNodes = resps.vs.fNodes;
            const auto& mNodes = resps.vs.mNodes;
            computeDigest(&lHash[i - fStart], fNodes.pt[i].x, fNodes.pt[i].y, fNodes.pt[i].z, fNodes.removal[i]);

            K key;
            bool isRC;
            decodeParent(fNodes.parentCode[i], key, isRC);

            auto v = table.Find(key);
            //assert(v.has_value());
            size_t midx = *v - mStart;
            childHash[midx][isRC] = &lHash[i - fStart];

            while (visitCount[midx].cnt++ == 1) {
                int k = idx;
                //assert(childHash[midx][0] != nullptr);
                //assert(childHash[midx][1] != nullptr);
                computeDigest(&mHash[midx], childHash[midx][0], childHash[midx][1],
                    mNodes.splitDim[midx + mStart],
                    mNodes.splitVal[midx + mStart],
                    mNodes.removal[midx + mStart]);

                int parentCode = mNodes.parentCode[midx + mStart];
                if (parentCode == -1) {
                    rootIndex = midx;
                    break;
                }
                decodeParent(mNodes.parentCode[midx + mStart], key, isRC);
                v = table.Find(key);
                //assert(v.has_value());
                size_t next = *v - mStart;
                childHash[next][isRC] = &mHash[midx];
                midx = next;
            }
        }

        // clean up
        table.clear();

        return rootIndex < 0 || equal(mHash[rootIndex], rootHash);
    }
}