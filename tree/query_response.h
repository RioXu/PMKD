#pragma once
#include <common/geometry/aabb.h>

namespace pmkd {
	using Query = vec3f;
	using RangeQuery = AABB;

	const uint32_t DEFAULT_MAX_SIZE_PER_RANGE_RESPONSE = 30;


	struct QueryResponse {
		bool exist = false;
	};

	struct RangeQueryResponse {
		vec3f* pts = nullptr;
		uint32_t* size = nullptr;
	};

	struct RangeQueryResponsesRawRepr {
		vec3f* buffer;
		uint32_t* respSize;
		const uint32_t capPerResponse;

		vec3f* getBufPtr(uint32_t idx) { return buffer + idx * capPerResponse; }
		uint32_t* getSizePtr(uint32_t idx) { return respSize + idx; }
	};

	struct RangeQueryResponses {
		vector<vec3f> buffer;
		vector<uint32_t> respSize;
		uint32_t numResponse;
		uint32_t capPerResponse;

		RangeQueryResponses(uint32_t num, uint32_t capacityPerResponse = DEFAULT_MAX_SIZE_PER_RANGE_RESPONSE)
			:numResponse(num), capPerResponse(capacityPerResponse),
			buffer(num* capacityPerResponse), respSize(num, 0) {}

		RangeQueryResponses(const RangeQueryResponses&) = delete;
		RangeQueryResponses& operator=(const RangeQueryResponses&) = delete;

		RangeQueryResponses(RangeQueryResponses&& other) {
			buffer = std::move(other.buffer);
			respSize = std::move(other.respSize);
			numResponse = other.numResponse;
			capPerResponse = other.capPerResponse;
		}

		RangeQueryResponses& operator=(RangeQueryResponses&& other) {
			buffer = std::move(other.buffer);
			respSize = std::move(other.respSize);
			numResponse = other.numResponse;
			capPerResponse = other.capPerResponse;
			return *this;
		}

		~RangeQueryResponses() {}

		void reconfig(uint32_t num, uint32_t capacityPerResponse) {
			numResponse = num;
			capPerResponse = capacityPerResponse;
			buffer.clear();
			buffer.resize(numResponse * capPerResponse);
			respSize.resize(numResponse, 0);
		}

		size_t size() const { return respSize.size(); }

		RangeQueryResponse at(uint32_t idx) {
			return { buffer.data() + idx * capPerResponse,respSize.data() + idx };
		}

		RangeQueryResponsesRawRepr getRawRepr() const {
			return RangeQueryResponsesRawRepr{
				const_cast<vec3f*>(buffer.data()),
				const_cast<uint32_t*>(respSize.data()),
				capPerResponse
			};
		}
	};
}