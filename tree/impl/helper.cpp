#include <tree/helper.h>

namespace pmkd {
    void compare_vectors(const float* vec1, const float* vec2, int* result) {
        // Load the first vector
        __m256 v1 = _mm256_loadu_ps(vec1);
        // Load the second vector
        __m256 v2 = _mm256_loadu_ps(vec2);

        // Compare the vectors (e.g., _MM_CMPINT_GE for ">=")
        __m256 cmp_result = _mm256_cmp_ps(v1, v2, _MM_CMPINT_GE);

        // The result of the comparison is a vector of -1.0f or 0.0f
        // depending on the comparison result. Convert this to a vector
        // of 0.0f or 1.0f for a more conventional boolean result.
        __m256 bool_result = _mm256_and_ps(cmp_result, _mm256_set1_ps(1.0f));

        // Store the result back to memory
        _mm256_storeu_si256((__m256i*)result, (__m256i)bool_result);
    }
}