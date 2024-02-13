#pragma once
#include <cstdint>
#include <common/basic/vector.h>

namespace pmkd {
	template<int nbit>
	struct Morton;

	uint32_t inline leftShift3f_32b(uint32_t x) {
		if (x == (1 << 10)) --x;
		x = (x | (x << 16)) & 0b00000011000000000000000011111111;
		x = (x | (x << 8)) & 0b00000011000000001111000000001111;
		x = (x | (x << 4)) & 0b00000011000011000011000011000011;
		x = (x | (x << 2)) & 0b00001001001001001001001001001001;
		return x;
	}

	uint64_t inline leftShift3f_64b(uint32_t v) {
		uint64_t x = v & 0x1fffff;
		x = (x | x << 32) & 0x1f00000000ffff;
		x = (x | x << 16) & 0x1f0000ff0000ff;
		x = (x | x << 8) & 0x100f00f00f00f00f;
		x = (x | x << 4) & 0x10c30c30c30c30c3;
		x = (x | x << 2) & 0x1249249249249249;
		return x;
	}


	template<>
	struct Morton<32> {
		uint32_t code;

		Morton<32>() {}
		Morton<32>(uint32_t code) :code(code) {}

		Morton<32>& operator=(const Morton<32>& cls) {
			code = cls.code;
			return *this;
		}

		static inline int length() { return 32; }

		static inline Morton<32> calculate(float x, float y, float z) {
			uint32_t ux = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
			uint32_t uy = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
			uint32_t uz = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
			return (leftShift3f_32b(ux) << 2) |
				(leftShift3f_32b(uy) << 1) |
				leftShift3f_32b(uz);
		}

		static inline Morton<32> calculate(double x, double y, double z) {
			uint32_t ux = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
			uint32_t uy = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
			uint32_t uz = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
			return (leftShift3f_32b(ux) << 2) |
				(leftShift3f_32b(uy) << 1) |
				leftShift3f_32b(uz);
		}

		static inline uint8_t calcMetric(const Morton<32>& mc1, const Morton<32>& mc2) {
			return 32 - clz32(mc1.code ^ mc2.code);
		}

		static inline void calcSplit(uint8_t metric, Morton<32> rm, const vec3f& ptMin, const vec3f& ptMax,
			int* splitDim, mfloat* splitVal) {

			metric = 32 - metric;
			int dim = metric % 3;
			*splitDim = dim;
			mfloat val = 0;
			mfloat x = 0.5;
			for (int i = 1; i <= metric / 3 + 1; ++i) {
				int idx = 3 * (i - 1) + dim;
				val += ((rm.code >> (31 - idx)) & 1) * x;
				x /= 2;
			}
			val = val * (ptMax[dim] - ptMin[dim]) + ptMin[dim];
			*splitVal = val;
		}

		static inline void calcSplit(uint8_t metric, const vec3f& pt, int* splitDim, mfloat* splitVal) {
			int dim = (32 - metric) % 3;
			*splitDim = dim;
			*splitVal = pt[dim];
		}
	};

	template<>
	struct Morton<64> {
		using value_type = uint64_t;
		uint64_t code;

		Morton<64>() {}
		Morton<64>(uint64_t code) : code(code) {}

		Morton<64>& operator=(const Morton<64>& cls) {
			code = cls.code;
			return *this;
		}

		static constexpr int length() { return 64; }

		static inline Morton<64> calculate(float x, float y, float z) {
			uint32_t ux = fmin(fmax(x * 2097152.0f, 0.0f), 2097151.0f);
			uint32_t uy = fmin(fmax(y * 2097152.0f, 0.0f), 2097151.0f);
			uint32_t uz = fmin(fmax(z * 2097152.0f, 0.0f), 2097151.0f);
			return (leftShift3f_64b(ux) << 2) |
				(leftShift3f_64b(uy) << 1) |
				leftShift3f_64b(uz);
		}

		static inline Morton<64> calculate(double x, double y, double z) {
			uint32_t ux = fmin(fmax(x * 2097152.0, 0.0), 2097151.0);
			uint32_t uy = fmin(fmax(y * 2097152.0, 0.0), 2097151.0);
			uint32_t uz = fmin(fmax(z * 2097152.0, 0.0), 2097151.0);
			return (leftShift3f_64b(ux) << 2) |
				(leftShift3f_64b(uy) << 1) |
				leftShift3f_64b(uz);
		}

		static inline uint8_t calcMetric(const Morton<64>& mc1, const Morton<64>& mc2) {
			return 64 - clz64(mc1.code ^ mc2.code);
		}

		static inline void calcSplit(uint8_t metric, Morton<64> rm, const vec3f& ptMin, const vec3f& ptMax,
			int* splitDim, mfloat* splitVal) {

			metric = 64 - metric - 1;
			int dim = metric % 3;
			*splitDim = dim;
			mfloat val = 0;
			mfloat x = 0.5;
			for (int i = 1; i <= metric / 3 + 1; ++i) {
				int idx = 3 * (i - 1) + dim;
				val += ((rm.code >> (62 - idx)) & 1) * x;
				x /= 2;
			}
			val = val * (ptMax[dim] - ptMin[dim]) + ptMin[dim];
			*splitVal = val;
		}

		static inline void calcSplit(uint8_t metric, const vec3f& pt, int* splitDim, mfloat* splitVal) {
			int dim = (64 - metric) % 3;
			*splitDim = dim;
			*splitVal = pt[dim];
		}
	};
}