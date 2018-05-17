#pragma once

#include "pool.h"

class TPartitioning {
public:
    TPartitioning(const TPool& pool, const std::vector<size_t>& ids, size_t featureId, size_t splits);
    float GetSplitGain(size_t splitId, size_t minLeafSize) const;

private:
    std::vector<float> Sums;
    std::vector<size_t> Counts;
    size_t Size = 0;
};