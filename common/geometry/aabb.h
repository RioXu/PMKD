#pragma once
#include <string>
#include <common/basic/vector.h>

namespace pmkd {
	class AABB {
	public:
		AABB() { reset(); }
		AABB(const AABB& b) { ptMin = b.ptMin; ptMax = b.ptMax; }
		AABB(const vec3f& v) { ptMin = v; ptMax = v; }
		AABB(const vec3f& vMin, const vec3f& vMax) { ptMin = vMin; ptMax = vMax; }
		AABB(mfloat minx, mfloat miny, mfloat minz,
			mfloat maxx, mfloat maxy, mfloat maxz)
			:ptMin(vec3f(minx, miny, minz)),
			ptMax(vec3f(maxx, maxy, maxz)) {}

		inline vec3f center() const { return (ptMin + ptMax) * 0.5f; }

		inline void merge(const AABB& b) {
			ptMin = vec3f(fmin(ptMin.x, b.ptMin.x), fmin(ptMin.y, b.ptMin.y), fmin(ptMin.z, b.ptMin.z));
			ptMax = vec3f(fmax(ptMax.x, b.ptMax.x), fmax(ptMax.y, b.ptMax.y), fmax(ptMax.z, b.ptMax.z));
		}

		inline void merge(const vec3f& b) {
			ptMin = vec3f(fmin(ptMin.x, b.x), fmin(ptMin.y, b.y), fmin(ptMin.z, b.z));
			ptMax = vec3f(fmax(ptMax.x, b.x), fmax(ptMax.y, b.y), fmax(ptMax.z, b.z));
		}

		inline void merge(const mfloat x, const mfloat y, const mfloat z) {
			ptMin = vec3f(fmin(ptMin.x, x), fmin(ptMin.y, y), fmin(ptMin.z, z));
			ptMax = vec3f(fmax(ptMax.x, x), fmax(ptMax.y, y), fmax(ptMax.z, z));
		}

		inline bool overlap(const AABB& b) const {
			if (b.ptMin.x > ptMax.x || b.ptMax.x < ptMin.x) return false;
			if (b.ptMin.y > ptMax.y || b.ptMax.y < ptMin.y) return false;
			if (b.ptMin.z > ptMax.z || b.ptMax.z < ptMin.z) return false;
			return true;
		}

		inline bool include(const vec3f& pt) const {
			return ptMin.x <= pt.x && pt.x <= ptMax.x &&
				ptMin.y <= pt.y && pt.y <= ptMax.y &&
				ptMin.z <= pt.z && pt.z <= ptMax.z;
		}

		inline bool include(const AABB& b) const {
			return include(b.ptMin) && include(b.ptMax);
		}

		bool operator==(const AABB& b) const { return ptMin == b.ptMin && ptMax == b.ptMax; }

		inline void reset() {
			ptMin = vec3f(FMAX, FMAX, FMAX);
			ptMax = vec3f(-FMAX, -FMAX, -FMAX);
		}

		std::string toString() const {
			char content[50];
#if USE_DOUBLE_PRECISION
			sprintf(content, "{(%.2lf,%.2lf,%.2lf), (%.2lf,%.2lf,%.2lf)}", ptMin.x, ptMin.y, ptMin.z, ptMax.x, ptMax.y, ptMax.z);
#else
			sprintf(content, "{(%.2f,%.2f,%.2f), (%.2f,%.2f,%.2f)}", ptMin.x, ptMin.y, ptMin.z, ptMax.x, ptMax.y, ptMax.z);
#endif
			return std::string(content);
		}

		vec3f ptMin, ptMax;
	};

	// self incremental binary op on AABB
	struct MergeOp {
		void operator() (AABB& box, const AABB& b) {
			box.merge(b);
		}

		void operator() (AABB& box, const vec3f& b) {
			box.merge(b);
		}

		void operator() (AABB& box, const mfloat x, const mfloat y, const mfloat z) {
			box.merge(x, y, z);
		}
	};
}