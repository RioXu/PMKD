#pragma once
#include <parlay/sequence.h>
#include <vector>
#include <memory>

#include "node.h"
#include "query_response.h"

namespace pmkd {

	struct PMKD_Config {
		AABB globalBoundary;
		bool optimize = false;
		uint32_t rebuildOnlyThreshold = 1e5;
		float expandFactor = 1.5f;
	};

	struct PMKD_PrintInfo;

	class PMKDTree {
	private:
		vector<vec3f> pts;
		AABB sceneBoundary;
		AABB globalBoundary;

		Leaves leaves;
		Interiors interiors;
		size_t ptNum;

		class BufferPool;

		std::unique_ptr<BufferPool> bufferPool;

		PMKD_Config config;
	public:
		PMKDTree();

		PMKDTree(const PMKD_Config& config);

		~PMKDTree();

		//void setConfig(const PMKD_Config& config);

		PMKD_PrintInfo print(bool verbose = false) const;

		void destroy();

		size_t primSize() const { return ptNum; }

		size_t primCapacity() const { return pts.capacity(); }

		std::vector<QueryResponse> query(const vector<Query>& queries) const;

		RangeQueryResponses query(const vector<RangeQuery>& queries) const;

		void insert(const vector<vec3f>& pts);

		void insert(vector<vec3f>&& pts);

		void remove(const vector<vec3f>& pts);

		// mixed operations
		void execute() {}

	private:
		void init();

		void buildStatic();

		void buildIncrement(size_t offset);

		void expandStorage(size_t newCapacity);
	};

	struct PMKD_PrintInfo {
		// leaf index transformed to idx + leafNum
		// interior index unchanged
		std::vector<int> preorderTraversal, inorderTraversal;
		std::vector<MortonType> leafMortons;
		std::vector<int> metrics;  // interior
		size_t leafNum;

		// verbose
		std::vector<int> splitDim;      // interior
		std::vector<mfloat> splitVal;   // interior  
		std::vector<vec3f> leafPoints;

		PMKD_PrintInfo() {}
		PMKD_PrintInfo(const PMKD_PrintInfo&) = default;
		PMKD_PrintInfo(PMKD_PrintInfo&&) = default;
	};
}