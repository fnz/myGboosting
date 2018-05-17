#include <iostream>
#include <numeric>
#include "partitioning.h"

TPartitioning::TPartitioning(const TSplits& splits) {
    for (const auto& split : splits) {
        Sums.emplace_back(1 + split.size(), 0.0);
        Counts.emplace_back(1 + split.size(), 0);
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

void TPartitioning::BuildFromIds(const std::vector<size_t>& ids, const TPool& pool, const std::vector<bool>& used, bool full) {
    Size = ids.size();
    auto featureCount = Sums.size();

    if (full) {
        #pragma omp parallel for
        for (size_t featureId = 0; featureId < featureCount; featureId++) {
            if (used[featureId]) {
                continue;
            }

            const auto& feature = pool.Features[featureId];

            auto& s = Sums[featureId];
            auto& c = Counts[featureId];

            for (size_t i = 0; i < ids.size(); ++i) {
                auto bin = feature[i];
                s[bin] += pool.Target[i];
                c[bin]++;
            }
        }
    } else {
        TTarget target(ids.size(), 0);
        for (size_t i = 0; i < ids.size(); i++) {
            target[i] = pool.Target[ids[i]];
        }

        #pragma omp parallel for
        for (size_t featureId = 0; featureId < featureCount; featureId++) {
            if (used[featureId]) {
                continue;
            }

            const auto& feature = pool.Features[featureId];
            auto& s = Sums[featureId];
            auto& c = Counts[featureId];

            for (size_t i = 0; i < ids.size(); ++i) {
                auto bin = feature[ids[i]];
                s[bin] += target[i];
                c[bin]++;
            }
        }
    }

    #pragma omp parallel for
    for (size_t featureId = 0; featureId < featureCount; featureId++) {
        auto& s = Sums[featureId];
        auto& c = Counts[featureId];
        std::partial_sum(s.begin(), s.end(), s.begin());
        std::partial_sum(c.begin(), c.end(), c.begin());
    }
}

void TPartitioning::BuildFromRelatives(const TPartitioning& parent, const TPartitioning& sibling, const std::vector<bool>& used) {
    Size = parent.Size - sibling.Size;
    auto featureCount = Sums.size();

    #pragma omp parallel for
    for (size_t featureId = 0; featureId < featureCount; featureId++) {
        auto& s = Sums[featureId];
        auto& c = Counts[featureId];
        for (size_t i = 0; i < s.size(); i++) {
            s[i] = parent.Sums[featureId][i] - sibling.Sums[featureId][i];
            c[i] = parent.Counts[featureId][i] - sibling.Counts[featureId][i];
        }
    }
}
