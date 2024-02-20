#pragma once
#include <cstdint>
#include <cmath>

namespace pmkd {
    struct SHA192 {
        uint8_t byte[24];
    };

    // device functions
    // leaf node
    inline void computeDigest(SHA192* digest, float x, float y, float z, bool removed=false) {
        digest->byte[0] = static_cast<uint32_t>(std::abs(x*1024)) & 0xff;
        digest->byte[1] = static_cast<uint32_t>(std::abs(y * 1024)) & 0xff;
        digest->byte[2] = static_cast<uint32_t>(std::abs(z * 1024)) & 0xff;
        digest->byte[3] = removed;
    }

    inline void computeDigest(SHA192* digest, double x, double y, double z, bool removed=false) {
        digest->byte[0] = static_cast<uint64_t>(std::abs(x * 2097152)) & 0xff;
        digest->byte[1] = static_cast<uint64_t>(std::abs(y * 2097152)) & 0xff;
        digest->byte[2] = static_cast<uint64_t>(std::abs(z * 2097152)) & 0xff;
        digest->byte[3] = removed;
    }

    // interior node
    inline void computeDigest(SHA192* digest, const SHA192* lcDigest, const SHA192* rcDigest, int dimension, float value) {
        digest->byte[0] = lcDigest->byte[0] ^ rcDigest->byte[0];
        digest->byte[1] = dimension ^ (static_cast<uint32_t>(std::abs(value * 1024)) & 0xff);
        digest->byte[2] = lcDigest->byte[2] ^ rcDigest->byte[2];
    }

    inline void computeDigest(SHA192* digest, const SHA192* lcDigest, const SHA192* rcDigest, int dimension, double value) {
        digest->byte[0] = lcDigest->byte[0] ^ rcDigest->byte[0];
        digest->byte[1] = dimension ^ (static_cast<uint32_t>(std::abs(value * 2097152)) & 0xff);
        digest->byte[2] = lcDigest->byte[2] ^ rcDigest->byte[2];
    }
}