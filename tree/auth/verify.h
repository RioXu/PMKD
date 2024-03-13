#pragma once
#include <parlay_hash/unordered_map.h>
#include <query_response.h>

namespace pmkd {
    bool verifyRangeQuery(const hash_t& rootHash, const RangeQuery& query, const VerifiableRangeQueryResponses& resps, size_t idx,
        parlay::parlay_unordered_map<int, size_t>& table);

    bool verifyRangeQuery_Sequential(const hash_t& rootHash, const RangeQuery& query, const VerifiableRangeQueryResponses& resps, size_t idx,
        parlay::parlay_unordered_map<int, size_t>& table);
}