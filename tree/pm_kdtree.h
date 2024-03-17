#pragma once
#include <queue>
#include <vector>
#include <memory>
#include <parlay/sequence.h>

#include <node.h>
#include <query_response.h>

namespace pmkd {

	struct PMKD_Config {
		AABB globalBoundary;
		bool optimize = true;
		// rebuild conditions
		int maxNumBatches = 20;
		float maxRemovedRatio = 1.0f; // total removed / total valid before this removal
		float maxDInsertedRatio = 1.0f; // total dynamically inserted / total valid before this insertion
	};

	struct PMKD_PrintInfo;

	class PMKDTree {
#ifdef M_DEBUG
	public:
#else
	private:
#endif
		//AABB sceneBoundary;
		AABB globalBoundary;

		std::unique_ptr<NodeMgr> nodeMgr;

		class BufferPool;

		std::unique_ptr<BufferPool> bufferPool;

		PMKD_Config config;

		bool needRebuild(int nToDInsert, int nToRemove) const;

		// status
		bool isStatic;
		int nTotalRemoved;
		int nTotalDInserted;
	public:
		PMKDTree();

		PMKDTree(const PMKD_Config& config);

		~PMKDTree();

		//void setConfig(const PMKD_Config& config);

		PMKD_PrintInfo print(bool verbose = false) const;

		void destroy();

		AABB getGlobalBoundary() const { return globalBoundary; }

		size_t primSize() const { return nodeMgr->numLeaves(); }

		std::vector<vec3f> getStoredPoints() const;

		QueryResponses query(const vector<Query>& queries) const;

		RangeQueryResponses query(const vector<RangeQuery>& queries) const;

#ifdef ENABLE_MERKLE
		VerifiableRangeQueryResponses
			verifiableQuery(const vector<RangeQuery>& queries) const;

		hash_t getRootHash() const;
#endif

		void findBin_Experiment(const vector<vec3f>& pts, int version, bool print = false);

		void insert(const vector<vec3f>& ptsAdd);

		void insert_v2(const vector<vec3f>& ptsAdd);

		void firstInsert(const vector<vec3f>& ptsAdd);

		void remove(const vector<vec3f>& ptsRemove);

		void remove_v2(const vector<vec3f>& ptsRemove);

		// mixed operations
		void execute(const vector<vec3f>& ptsRemove, const vector<vec3f>& ptsAdd);

	private:
		void init();

		void sortPts(const vector<vec3f>& pts, vector<vec3f>& ptsSorted) const;
		void sortPts(const vector<vec3f>& pts, vector<vec3f>& ptsSorted, vector<int>& primIdxInited) const;
		void sortPts(const vector<vec3f>& pts, vector<vec3f>& ptsSorted, vector<int>& primIdx, vector<MortonType>& mortons) const;

		void rebuildUponInsert(const vector<vec3f>& ptsAdd);

		void rebuildUponRemove(const vector<vec3f>& ptsRemove);

		void buildStatic(const vector<vec3f>& pts);

		void buildStatic_LeavesReady(Leaves& leaves, Interiors& interiors);

		void buildIncrement(const vector<vec3f>& ptsAdd);

		void buildIncrement_v2(const vector<vec3f>& ptsAdd);

		void _query(const vector<RangeQuery>& queries, RangeQueryResponses& responses) const;

		PMKD_PrintInfo printStatic(bool verbose) const;

		PMKD_PrintInfo printDynamic(bool verbose) const;
	};

	template<typename T>
	struct CustomLess {
		bool operator()(const vector<T>& a, const vector<T>& b) const {
            return a.capacity() < b.capacity();
        }
	};
	// BufferPool
	class PMKDTree::BufferPool {
	private:
		template<typename T>
		using buffers_t = std::priority_queue < vector<T>, std::vector<vector<T>>, CustomLess<T>>;

		buffers_t<uint8_t> byteBuffers;
		buffers_t<int> intBuffers;
		buffers_t<mfloat> floatBuffers;
		buffers_t<vec3f> vec3fBuffers;
		buffers_t<MortonType> mortonBuffers;

		template<typename T>
		buffers_t<T>& getDeque();

		template<>
		buffers_t<uint8_t>& getDeque<uint8_t>() { return byteBuffers; }

		template<>
		buffers_t<int>& getDeque<int>() { return intBuffers; }

		template<>
		buffers_t<mfloat>& getDeque<mfloat>() { return floatBuffers; }

		template<>
		buffers_t<vec3f>& getDeque<vec3f>() { return vec3fBuffers; }

		template<>
		buffers_t<MortonType>& getDeque<MortonType>() { return mortonBuffers; }
	public:
		BufferPool() {}
		~BufferPool() {}

		template<typename T>
		vector<T> acquire(size_t size) {
			auto& dq = getDeque<T>();
			if (dq.empty()) { return vector<T>(size); }
			//auto buffer = std::move(dq.front());
			//auto& buffer = dq.front();
			//auto buffer = dq.front();
			vector<T> buffer(std::move(dq.top()));
			//auto buffer(dq.front());
			dq.pop();

			if (buffer.size() != size)
				buffer.resize(size);
			return std::move(buffer);
		}

		template<typename T>
		vector<T> acquire(size_t size, T val) {
			auto& dq = getDeque<T>();
			if (dq.empty()) { return vector<T>(size, val); }

			vector<T> buffer(std::move(dq.top()));
			dq.pop();

			buffer.clear();
			buffer.resize(size, val);
			return std::move(buffer);
		}

		template<typename T>
		void release(vector<T>&& buffer) {
			static_assert(std::is_rvalue_reference_v<decltype(buffer)>);

			if (buffer.empty()) return;
			auto& dq = getDeque<T>();
			dq.push(std::move(buffer));
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
		PMKD_PrintInfo(const PMKD_PrintInfo&) = delete;
		PMKD_PrintInfo(PMKD_PrintInfo&&) = default;
	};
}