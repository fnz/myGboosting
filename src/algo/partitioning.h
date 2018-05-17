#pragma once

#include "pool.h"

class TPartitioning {
public:
    TPartitioning(const TSplits& splits);
    float GetSplitGain(size_t featureId, size_t splitId, size_t minLeafSize) const;

    void BuildFromIds(const std::vector<size_t>& ids, const TPool& pool, const std::vector<bool>& used, bool full = false);
    void BuildFromRelatives(const TPartitioning& parent, const TPartitioning& sibling, const std::vector<bool>& used);

private:
    std::vector<std::vector<float>> Sums;
    std::vector<std::vector<size_t>> Counts;
    size_t Size = 0;
};