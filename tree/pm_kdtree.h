#pragma once
#include <deque>
#include <vector>
#include <memory>
#include <parlay/sequence.h>

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

		AABB getGlobalBoundary() const { return globalBoundary; }

		size_t primSize() const { return ptNum; }

		size_t primCapacity() const { return pts.capacity(); }

		QueryResponses query(const vector<Query>& queries) const;

		RangeQueryResponses query(const vector<RangeQuery>& queries) const;

		void findBin_Experiment(const vector<vec3f>& pts, int version);

		void insert(const vector<vec3f>& ptsAdd);

		template<typename Range>
		void firstInsert(Range&& range) {
			ptNum += range.size();
			this->pts = range;
			buildStatic();
		}

		void remove(const vector<vec3f>& ptsRemove);

		// mixed operations
		void execute() {}

	private:
		void init();

		void buildStatic();

		void buildIncrement(const vector<vec3f>& ptsAdd);

		void expandStorage(size_t newCapacity);
	};

	// BufferPool
	class PMKDTree::BufferPool {
	private:
		std::deque<vector<int>> intBuffers;
		std::deque<vector<mfloat>> floatBuffers;
		std::deque<vector<vec3f>> vec3fBuffers;
		std::deque<vector<MortonType>> mortonBuffers;

		template<typename T>
		std::deque<vector<T>>& getDeque();

		template<>
		std::deque<vector<int>>& getDeque<int>() { return intBuffers; }

		template<>
		std::deque<vector<mfloat>>& getDeque<mfloat>() { return floatBuffers; }

		template<>
		std::deque<vector<vec3f>>& getDeque<vec3f>() { return vec3fBuffers; }

		template<>
		std::deque<vector<MortonType>>& getDeque<MortonType>() { return mortonBuffers; }
	public:
		BufferPool() {}
		~BufferPool() {}

		template<typename T>
		vector<T> acquire(size_t size) {
			auto& dq = getDeque<T>();
			if (dq.empty()) { return vector<T>(size); }
			auto& buffer = dq.front();
			dq.pop_front();

			buffer.resize(size);
			return std::move(buffer);
		}

		template<typename T>
		void release(vector<T>&& buffer) {
			static_assert(std::is_rvalue_reference_v<decltype(buffer)>);

			if (buffer.empty()) return;
			auto& dq = getDeque<T>();
			dq.push_back(buffer);
		}
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