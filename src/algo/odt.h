#pragma once

#include "defines.h"
#include "split.h"
#include "pool.h"

#include <iostream>

class TPartitioning {
public:
    TPartitioning(size_t featureId, const std::vector<size_t>& ids, const TPool& pool, size_t splits)
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

    float GetSplitGain(size_t splitId, size_t minLeafSize) const {
        float leftCount = Counts[splitId];
        if (leftCount < minLeafSize) {
            return 0.0;
        }

        float rightCount = Counts.back() - leftCount;
        if (rightCount < minLeafSize) {
            return 0.0;
        }

        float leftSum = Sums[splitId];
        float leftGain = leftSum*leftSum/leftCount;

        float rightSum = Sums.back() - leftSum;
        float rightGain = rightSum*rightSum/rightCount;

        return (leftGain + rightGain) / float(Size);
    }

private:
    std::vector<float> Sums;
    std::vector<size_t> Counts;
    size_t Size = 0;
};

class TObliviousDecisionTree {
public:
    static TObliviousDecisionTree
    Fit(const TPool& pool, const TSplits& splits, size_t maxDepth, size_t minCount, float sampleRate);

    float Predict(const TFeatureRow& data) const;

public:
    std::vector<size_t> Features;
    std::vector<size_t> Splits;
    std::vector<float> Values;
};
