#include "odt.h"

#include "partitioning.h"

#include <vector>
#include <iostream>
#include <omp.h>

static float getPrediction(size_t partId, const TPool& pool, const std::vector<std::vector<size_t>>& parts) {
    if (parts[partId].empty()) {
        return getPrediction(partId / 2, pool, parts);
    } else {
        float sum = 0.0;
        for (size_t id : parts[partId]) {
            sum += pool.Target[id];
        }
        sum /= float(parts[partId].size());
        return sum;
    }
}

TObliviousDecisionTree
TObliviousDecisionTree::Fit(const TPool& pool, const TSplits& splits, size_t maxDepth, size_t minCount,
                            float sampleRate) {
    TObliviousDecisionTree tree;
    auto size = (size_t(1) << (maxDepth + 1));
    std::vector<std::vector<size_t>> parts(size);
    std::vector<bool> used(pool.BinarizedFeatureCount, false);

    std::vector<TPartitioning> ps;
    ps.reserve(size);
    TPartitioning partitioning(splits);
    for (size_t i = 0; i < size; i++) {
        ps.push_back(partitioning);
    }

    std::vector<size_t> ids;
    if (sampleRate == 1.0) {
        for (size_t i = 0; i < pool.Size; i++) {
            ids.push_back(i);
        }
    } else {
        for (size_t i = 0; i < pool.Size; i++) {
            float coin = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);;
            if (coin < sampleRate) {
                ids.push_back(i);
            }
        }
    }

    parts[0] = std::move(ids);
    ps[0].BuildFromIds(parts[0], pool, used, sampleRate == 1.0);

    size_t firstPart = 0;
    size_t lastPart = 0;

    size_t depth = 1;

    omp_lock_t lock;
    omp_init_lock(&lock);

    for (; depth <= maxDepth; depth++) {
        firstPart = (size_t(1) << (depth - 1)) - 1;
        lastPart = (size_t(1) << depth) - 2;

        float maxGain = 0.0;
        size_t bestFeature = 0;
        size_t bestSplit = 0;

        #pragma omp parallel for
        for (size_t featureId = 0; featureId < pool.BinarizedFeatureCount; featureId++) {
            if (used[featureId]) {
                continue;
            }

            for (size_t splitId = 0; splitId < splits[featureId].size(); splitId++) {
                float gain = 0.0;
                for (size_t partId = firstPart; partId <= lastPart; partId++) {
                    gain += ps[partId].GetSplitGain(featureId, splitId, minCount);
                }

//                omp_set_lock(&lock);
                if (gain > maxGain) {
                    maxGain = gain;
                    bestFeature = featureId;
                    bestSplit = splitId;
                }
//                omp_unset_lock(&lock);
            }
        }

        if (maxGain == 0.0) {
            break;
        }

        tree.Features.push_back(bestFeature);
        tree.Splits.push_back(bestSplit);
        used[bestFeature] = true;

        #pragma omp parallel for
        for (size_t parentPartId = firstPart; parentPartId <= lastPart; parentPartId++) {
            size_t leftPartId = 2 * parentPartId + 1;
            size_t rightPartId = 2 * parentPartId + 2;

            parts[leftPartId].reserve(parts[parentPartId].size());
            parts[rightPartId].reserve(parts[parentPartId].size());

            for (size_t id : parts[parentPartId]) {
                if (pool.Features[bestFeature][id] <= bestSplit) {
                    parts[leftPartId].push_back(id);
                } else {
                    parts[rightPartId].push_back(id);
                }
            }

            if (depth != maxDepth) {
                size_t minPartId;
                size_t maxPartId;
                if (parts[leftPartId].empty()) {
                    ps[rightPartId] = ps[parentPartId];
                } else if (parts[rightPartId].empty()) {
                    ps[leftPartId] = ps[parentPartId];
                } else {}
                if (parts[leftPartId].size() < parts[rightPartId].size()) {
                    minPartId = leftPartId;
                    maxPartId = rightPartId;
                } else {
                    minPartId = rightPartId;
                    maxPartId = leftPartId;
                }
                ps[minPartId].BuildFromIds(parts[minPartId], pool, used);
                ps[maxPartId].BuildFromRelatives(ps[parentPartId], ps[minPartId], used);
            }

        }
    }

    omp_destroy_lock(&lock);

    tree.Values.resize(size_t(1) << depth);

    firstPart = (size_t(1) << (depth - 1)) - 1;
    lastPart = (size_t(1) << depth) - 2;

    #pragma omp parallel for
    for (size_t partId = firstPart; partId <= lastPart; partId++) {
        size_t valueId = partId - firstPart;
        tree.Values[valueId] = getPrediction(partId, pool, parts);
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






