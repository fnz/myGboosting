#include <iostream>
#include <numeric>
#include "partitioning.h"

TPartitioning::TPartitioning(const TPool& pool, const std::vector<size_t>& ids, size_t featureId, size_t splits)
        : Sums(1 + splits, 0.0)
        , Counts(1 + splits, 0)
{
    Size = ids.size();
    for (size_t id : ids) {
        auto bin = pool.Features[featureId][id];
        Sums[bin] += pool.Target[id];
        Counts[bin]++;
    }

    for (size_t i = 1; i < Sums.size(); i++) {
        Sums[i] += Sums[i - 1];
        Counts[i] += Counts[i - 1];
    }
}

float TPartitioning::GetSplitGain(size_t featureId, size_t splitId, size_t minLeafSize) const {
    float leftCount = Counts[featureId][splitId];
    if (leftCount < minLeafSize) {
        return 0.0;
    }

    float rightCount = Counts[featureId].back() - leftCount;
    if (rightCount < minLeafSize) {
        return 0.0;
    }

    float leftSum = Sums[featureId][splitId];
    float leftGain = leftSum*leftSum/leftCount;

    float rightSum = Sums[featureId].back()  - leftSum;
    float rightGain = rightSum*rightSum/rightCount;

    return (leftGain + rightGain) / float(Size);
}
