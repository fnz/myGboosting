#pragma once

#include "defines.h"
#include "pool.h"

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
