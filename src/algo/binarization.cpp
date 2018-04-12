#include "binarization.h"
#include "defines.h"
#include "histogram.h"

#include <vector>
#include <iostream>

std::vector<float> GetSplits(const TRawFeature& data, size_t parts) {
    return BuildBinBounds(data, parts);
}

TPool TBinarizer::Binarize(TRawPool&& raw) {
    TPool pool;
    pool.Names = std::move(raw.Names);
    pool.Target = std::move(raw.Target);
    pool.Size = raw.RawFeatures[0].size();

    Hashes = std::move(raw.Hashes);

    auto rawFeatureCount = raw.RawFeatures.size();

    for (size_t rawFeatureId = 0; rawFeatureId < rawFeatureCount; ++rawFeatureId) {
        TFeatures binarized;
        const auto& rawColumn = raw.RawFeatures[rawFeatureId];
        if (!Hashes[rawFeatureId].empty()) {
            binarized = BinarizeCatFeature(rawColumn, Hashes[rawFeatureId].size());
        } else {
            auto splits = GetSplits(raw.RawFeatures[rawFeatureId], 10);
            binarized = BinarizeFloatFeature(rawColumn, splits);
            Splits.emplace_back(std::move(splits));
        }

        for (auto& column : binarized) {
            pool.Features.emplace_back(std::move(column));
            BinarizedToRaw.push_back(rawFeatureId);
        }
    }

    pool.RawFeatureCount = raw.RawFeatures.size();
    pool.BinarizedFeatureCount = pool.Features.size();

    return pool;
}

TFeatureVector TBinarizer::Binarize(size_t featureId, const std::string& value) const {
    auto& hash = Hashes[featureId];
    auto it = hash.find(value);
    if (it == hash.end()) {
        throw std::logic_error("No such value for categorical feature was seen in the training pool");
    }

    TFeatureVector vector(hash.size());
    vector[it->second] = 1;
    return vector;
}

TFeatureVector TBinarizer::Binarize(size_t featureId, float value) const {
    const auto& splits = Splits[featureId];
    if (splits.empty()) {
        throw std::logic_error("No splits info found for this feature");
    }

    TFeatureVector vector(splits.size());
    for (size_t i = 0; i < splits.size(); ++i) {
        if (value >= splits[i]) {
            vector[i] = 1;
        }
    }
    return vector;
}

TFeatures TBinarizer::BinarizeFloatFeature(const TRawFeature& data, std::vector<float> splits) {
    size_t length = data.size();
    std::vector<std::vector<char>> binarized(splits.size(), std::vector<char>(length, 0));

    for (size_t i = 0; i < length; ++i) {
        for (size_t j = 0; j < splits.size(); ++j) {
            if (data[i] >= splits[j]) {
                binarized[j][i] = 1;
            }
        }
    }

    return binarized;
}

TFeatures TBinarizer::BinarizeCatFeature(const TRawFeature& data, size_t cats) {
    size_t length = data.size();
    std::vector<std::vector<char>> binarized(cats, std::vector<char>(length, 0));

    for (size_t i = 0; i < length; ++i) {
        binarized[size_t(data[i])][i] = 1;
    }

    return binarized;
}
