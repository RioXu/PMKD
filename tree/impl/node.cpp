#include <exception>
#include <tree/node.h>

namespace pmkd {
    void NodeMgr::append(Leaves&& leaves, Interiors&& interiors, vector<vec3f>&& pts, bool syncDevice) {
        if (interiors.rangeL.empty()) return;

        size_t nb = numBatches();
        int acc = leaves.size() + (nb > 0 ? sizesAcc[nb - 1] : 0);

        if (syncDevice) {
            dSizesAcc.push_back(acc);
            dLeavesBatch.push_back(leaves.getRawRepr());
            dInteriorsBatch.push_back(interiors.getRawRepr());
            dPtsBatch.push_back(pts.data());
        }

        sizesAcc.push_back(acc);
        leavesBatch.emplace_back(std::move(leaves));
        interiorsBatch.emplace_back(std::move(interiors));
        ptsBatch.emplace_back(std::move(pts));
    }

    NodeMgr::HostCopy NodeMgr::copyToHost() const {
        HostCopy nodeMgrH;
        nodeMgrH.interiorsBatch.reserve(numBatches());
        nodeMgrH.leavesBatch.reserve(numBatches());
        nodeMgrH.ptsBatch.reserve(numBatches());
        nodeMgrH.sizesAcc.reserve(numBatches());

        for (const auto& batch : leavesBatch) {
            nodeMgrH.leavesBatch.push_back(batch.copyToHost());
        }
        for (const auto& batch : interiorsBatch) {
            nodeMgrH.interiorsBatch.push_back(batch.copyToHost());
        }
        for (const auto& batch : ptsBatch) {
            nodeMgrH.ptsBatch.push_back(batch);
        }
        nodeMgrH.sizesAcc.insert(nodeMgrH.sizesAcc.end(), sizesAcc.begin(), sizesAcc.end());
        return nodeMgrH;
    }

    vector<vec3f> NodeMgr::flattenPoints() const {
        vector<vec3f> pts;
        pts.reserve(numLeaves());

        for (const auto& batch : ptsBatch) {
            pts.insert(pts.end(), batch.begin(), batch.end());
        }
        return pts;
    }

    void NodeMgr::syncDevice() {
        if (isDeviceSyncronized()) return;
        if (numBatches() == 0) {
            clearDevice();
            return;
        }
        // note: resize can be async
        dLeavesBatch.resize(leavesBatch.size());
        dInteriorsBatch.resize(interiorsBatch.size());
        dPtsBatch.resize(ptsBatch.size());
        dSizesAcc.resize(sizesAcc.size());

        vector<LeavesRawRepr> hLeavesBatch;
        hLeavesBatch.reserve(leavesBatch.size());
        for (size_t i = 0; i < leavesBatch.size(); i++) {
            hLeavesBatch.push_back(leavesBatch[i].getRawRepr());
        }
        vector<InteriorsRawRepr> hInteriorsBatch;
        hInteriorsBatch.reserve(interiorsBatch.size());
        for (size_t i = 0; i < interiorsBatch.size(); i++) {
            hInteriorsBatch.push_back(interiorsBatch[i].getRawRepr());
        }
        vector<vec3f*> hPtsBatch;
        hPtsBatch.reserve(ptsBatch.size());
        for (size_t i = 0; i < ptsBatch.size(); i++) {
            hPtsBatch.push_back(ptsBatch[i].data());
        }
        memcpy(dLeavesBatch.data(), hLeavesBatch.data(), sizeof(LeavesRawRepr) * leavesBatch.size());
        memcpy(dInteriorsBatch.data(), hInteriorsBatch.data(), sizeof(InteriorsRawRepr) * interiorsBatch.size());
        memcpy(dPtsBatch.data(), hPtsBatch.data(), sizeof(vec3f*) * ptsBatch.size());
        memcpy(dSizesAcc.data(), sizesAcc.data(), sizeof(int) * sizesAcc.size());
    }

    NodeMgrDevice NodeMgr::getDeviceHandle() const {
        if (!isDeviceSyncronized()) {
            throw std::runtime_error("Device is not syncronized");
        }
        NodeMgrDevice dNodeMgr;
        dNodeMgr.numBatches = leavesBatch.size();
        dNodeMgr.leavesBatch = const_cast<LeavesRawRepr*>(dLeavesBatch.data());
        dNodeMgr.interiorsBatch = const_cast<InteriorsRawRepr*>(dInteriorsBatch.data());
        dNodeMgr.ptsBatch = const_cast<vec3f**>(dPtsBatch.data());
        dNodeMgr.sizesAcc = const_cast<int*>(dSizesAcc.data());
        return dNodeMgr;
    }
}