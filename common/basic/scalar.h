#pragma once
#include <cmath>
#include <cfloat>

namespace pmkd {
#if USE_DOUBLE_PRECISION
	using mfloat = double;
#define FMAX DBL_MAX
#define FMIN DBL_MIN
#else
	using mfloat = float;
#define FMAX FLT_MAX
#define FMIN FLT_MIN
#endif

#ifdef _MSC_VER
#include <intrin.h>
#define clz32(x) __lzcnt(x)
#define clz64(x) __lzcnt64(x)
#elif (defined(__GNUC__) || defined(__clang__))
#include <stdint.h>
#define clz32(x) __builtin_clz(x)
#define clz64(x) __builtin_clzll(x)
#else
	uint32_t inline popcnt32(uint32_t x) {
		x -= ((x >> 1) & 0x55555555);
		x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
		x = (((x >> 4) + x) & 0x0f0f0f0f);
		x += (x >> 8);
		x += (x >> 16);
		return x & 0x0000003f;
	}

	uint64_t inline popcnt64(uint64_t x) {
		x -= ((x >> 1) & 0x5555555555555555);
		x = (((x >> 2) & 0x3333333333333333) + (x & 0x3333333333333333));
		x = (((x >> 4) + x) & 0x0f0f0f0f0f0f0f0f);
		x += (x >> 8);
		x += (x >> 16);
		x += (x >> 32);
		return x & 0x000000000000007f;
	}

	uint32_t inline clz32(uint32_t x) {
		x |= (x >> 1);
		x |= (x >> 2);
		x |= (x >> 4);
		x |= (x >> 8);
		x |= (x >> 16);
		return 32 - popcnt32(x);
	}

	uint64_t inline clz64(uint64_t x) {
		x |= (x >> 1);
		x |= (x >> 2);
		x |= (x >> 4);
		x |= (x >> 8);
		x |= (x >> 16);
		x |= (x >> 32);
		return 64 - popcnt64(x);
	}
#endif
	// possible optimization of x/2 using bitwise op
	// may not work or perform well on all platforms
	inline void div2(float &x) {
		int* intPtr = (int*)&x;
		*intPtr = *intPtr - (1 << 23);
	}

	inline void div2(double &x) {
		long long* intPtr = (long long*)&x;
		*intPtr = *intPtr - (1LL << 52);
	}
}