#pragma once
#include "node.h"

namespace pmkd {
    // inline void copyState(const BottomUpState& src, TopDownStates& dst, int idx) {
    //     uint8_t state = src.load(std::memory_order_relaxed);
    //     dst.child[0][idx] = state & 1;   // lc
    //     dst.child[1][idx] = state >> 1;  // rc
    // }

    // inline void copyState(const TopDownStates& src, BottomUpState& dst, int idx) {
    //     uint8_t state = src.child[0][idx] | (src.child[1][idx] << 1);
    //     dst.store(state, std::memory_order_relaxed);
    // }

    // inline void atomicCopyState(const TopDownStates& src, BottomUpState& dst, int idx, bool hasBeenCopied) {
    //     uint8_t newState = src.child[0][idx] | (src.child[1][idx] << 1);
    //     uint8_t oldState = dst.fetch_or(newState, std::memory_order_acq_rel);
    //     hasBeenCopied = oldState == 0;
    // }

    inline void setVisitStateTopDown(TopDownStates& states, int idx, bool toRC) {
        if (states.child[toRC][idx] != 1)
            states.child[toRC][idx] = 1;
    }

    // // return if visit state is cleared after modification
    inline bool unsetVisitStateBottomUp(TopDownStates& td, int idx, BottomUpState& bu, bool fromRC) {
        td.child[fromRC][idx] = 0;

        // td:01, bu: 0
        uint8_t otherVisited = td.child[1 - fromRC][idx];
        if (otherVisited == 0) return true;

        // td:11, bu: 0 or 1, if bu == 1 then move on
        uint8_t oldCnt = bu.fetch_add(1, std::memory_order_acq_rel) & 1;
        return oldCnt == 1;
    }

    inline void setRemoveStateBottomUp(BottomUpState& removeState, bool fromRC) {
        uint8_t bit = 1 << fromRC;
        removeState.fetch_or(bit, std::memory_order_acq_rel);
    }

    // return if remove state changed from not removed to removed
    // if true then move on
    inline bool setCheckRemoveStateBottomUp(BottomUpState& removeState, bool fromRC) {
        uint8_t bit = 1 << fromRC;
        return removeState.fetch_or(bit, std::memory_order_acq_rel) == (1 << (1 - fromRC));
    }

    // return if remove state changed from removed to not removed
    // if true then move on
    inline bool unsetRemoveStateBottomUp(BottomUpState& removeState, bool fromRC) {
        uint8_t bit = 1 << fromRC;
        return removeState.fetch_and(~bit, std::memory_order_acq_rel) == 0b11;
    }

    inline bool isInteriorRemoved(const BottomUpState& removeState) {
        uint8_t state = removeState.load(std::memory_order_relaxed);
        return state == 0b11;
    }

    
    inline bool setVisitCountBottomUp(BuildAid& aid, int idx, bool fromRC) {
        uint8_t oldCnt = aid.visitCount[idx].cnt.fetch_add(1, std::memory_order_acq_rel);
        return oldCnt == 1;
    }
}