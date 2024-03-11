#pragma once
#include <cstdint>
#include <cmath>
#include <string.h>
#include <openssl/sha.h>

namespace pmkd {
    struct sha256_t {
        uint8_t byte[SHA256_DIGEST_LENGTH];
    };

    inline bool equal(const sha256_t& a, const sha256_t& b) {
        return memcmp(a.byte, b.byte, SHA256_DIGEST_LENGTH) == 0;
    }

    // device functions
    // leaf node
    inline void computeDigest(sha256_t* digest, float x, float y, float z, bool removed = false) {
        uint8_t data[13];
        memcpy(data, &x, 4);
        memcpy(data + 4, &y, 4);
        memcpy(data + 8, &z, 4);
        data[12] = uint8_t(removed);

        SHA256(data, 13, digest->byte);
    }

    inline void computeDigest(sha256_t* digest, double x, double y, double z, bool removed=false) {
        uint8_t data[25];
        memcpy(data, &x, 8);
        memcpy(data + 8, &y, 8);
        memcpy(data + 16, &z, 8);
        data[24] = uint8_t(removed);

        SHA256(data, 25, digest->byte);
    }

    // interior node
    inline void computeDigest(sha256_t* digest, const sha256_t* lcDigest, const sha256_t* rcDigest, int dimension, float value, uint8_t removeState) {
        uint8_t data[2 * SHA256_DIGEST_LENGTH + 6];
        memcpy(data, lcDigest->byte, SHA256_DIGEST_LENGTH);
        memcpy(data + SHA256_DIGEST_LENGTH, rcDigest->byte, SHA256_DIGEST_LENGTH);
        memcpy(data + SHA256_DIGEST_LENGTH * 2, &value, 4);
        data[SHA256_DIGEST_LENGTH * 2 + 4] = uint8_t(dimension);
        data[SHA256_DIGEST_LENGTH * 2 + 5] = removeState;

        SHA256(data, 2 * SHA256_DIGEST_LENGTH + 6, digest->byte);
    }

    inline void computeDigest(sha256_t* digest, const sha256_t* lcDigest, const sha256_t* rcDigest, int dimension, double value, uint8_t removeState) {
        uint8_t data[2 * SHA256_DIGEST_LENGTH + 10];
        memcpy(data, lcDigest->byte, SHA256_DIGEST_LENGTH);
        memcpy(data + SHA256_DIGEST_LENGTH, rcDigest->byte, SHA256_DIGEST_LENGTH);
        memcpy(data + SHA256_DIGEST_LENGTH * 2, &value, 8);
        data[SHA256_DIGEST_LENGTH * 2 + 8] = uint8_t(dimension);
        data[SHA256_DIGEST_LENGTH * 2 + 9] = removeState;

        SHA256(data, 2 * SHA256_DIGEST_LENGTH + 10, digest->byte);
    }
}