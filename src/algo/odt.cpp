#include "odt.h"

#include <vector>

TObliviousDecisionTree TObliviousDecisionTree::Fit(const TPool& pool, const TSplits& splits, size_t maxDepth, size_t minCount, float sampleRate) {
    TObliviousDecisionTree tree;

    auto size = (size_t(1) << maxDepth);
    std::vector<std::vector<size_t>> parts(2*size);

    std::vector<size_t> ids;
    for (size_t i = 0; i < pool.Size; i++) {
        ids.push_back(i);
    }

    parts[0] = std::move(ids);

    size_t firstPart = 0;
    size_t lastPart = 0;

    size_t depth = 1;

    std::vector<bool> used(pool.BinarizedFeatureCount, false);
    for (; depth <= maxDepth; depth++) {

        firstPart = (size_t(1) << (depth - 1)) - 1;
        lastPart = (size_t(1) << depth) - 2;

        float maxGain = 0.0;
        size_t bestFeature = 0;
        size_t bestSplit = 0;

        for (size_t featureId = 0; featureId < pool.BinarizedFeatureCount; featureId++) {
            if (used[featureId]) {
                continue;
            }
            std::vector<TPartitioning> ps;
            ps.reserve((lastPart - firstPart + 1));
            for (size_t partId = firstPart; partId <= lastPart; partId++) {
                ps.emplace_back(featureId, parts[partId], pool, splits[featureId].size());
            }

            for (size_t splitId = 0; splitId < splits[featureId].size(); splitId++) {
                float gain = 0.0;
                for (const auto& partition : ps) {
                    gain += partition.GetSplitGain(splitId, minCount);
                }
                if (gain > maxGain) {
                    maxGain = gain;
                    bestFeature = featureId;
                    bestSplit = splitId;
                }
            }
        }

        if (maxGain == 0.0) {
            break;
        }

        tree.Features.push_back(bestFeature);
        tree.Splits.push_back(bestSplit);
        used[bestFeature] = true;

        for (size_t partId = firstPart; partId <= lastPart; partId++) {
            parts[2 * partId + 1].reserve(parts[partId].size());
            parts[2 * partId + 2].reserve(parts[partId].size());

            for (size_t id : parts[partId]) {
                if (pool.Features[bestFeature][id] <= bestSplit) {
                    parts[2 * partId + 1].push_back(id);
                } else {
                    parts[2 * partId + 2].push_back(id);
                }
            }
        }
    }

    tree.Values.resize(size_t(1) << depth);

    firstPart = (size_t(1) << (depth - 1)) - 1;
    lastPart = (size_t(1) << depth) - 2;

    for (size_t partId = firstPart; partId <= lastPart; partId++) {
        size_t valueId = partId - firstPart;
        for (size_t id : parts[partId]) {
            tree.Values[valueId] += pool.Target[id];
        }
        tree.Values[valueId] /= float(parts[partId].size());
    }

    return tree;
}

float TObliviousDecisionTree::Predict(const TFeatureRow& data) const {
    uint64_t mask = 0;
    for (size_t i = 0; i < Features.size(); i++) {
        mask *= 2;
        if (data[Features[i]] > Splits[i]) {
            mask++;
        }
    }
    return Values[mask];
}






