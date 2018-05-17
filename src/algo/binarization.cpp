#include "binarization.h"
#include "defines.h"

#include <algorithm>
#include <iostream>
#include <tgmath.h>
#include <vector>

std::vector<float> BuildSplits(const TRawFeature& data, size_t bins) {
    TRawFeature sorted(data);
    std::sort(sorted.begin(), sorted.end());

    std::vector<float> splits;
    auto binSize = size_t(ceil(data.size() / float(bins)));

    size_t currentBinSize = 1;

    for (size_t i = 1; i < sorted.size(); i++) {
        if (currentBinSize >= binSize && sorted[i] != sorted[i - 1]) {
            splits.push_back(float(0.5*(sorted[i] + sorted[i - 1])));
            currentBinSize = 0;
        }
        ++currentBinSize;
    }

    return splits;
}

TPool TBinarizer::Binarize(TRawPool&& raw, size_t maxBins) {
    Hashes = std::move(raw.Hashes);

    TPool pool;
    pool.Size = raw.RawFeatures[0].size();
    pool.Names = std::move(raw.Names);
    pool.Target = std::move(raw.Target);

    auto rawFeatureCount = raw.RawFeatures.size();

    for (size_t rawFeatureId = 0; rawFeatureId < rawFeatureCount; ++rawFeatureId) {
        const auto& rawColumn = raw.RawFeatures[rawFeatureId];
        if (!Hashes[rawFeatureId].empty()) {
            auto features = BinarizeCatFeature(rawColumn, Hashes[rawFeatureId].size());
            for (auto& feature : features) {
                pool.Features.emplace_back(std::move(feature));
                BinarizedToRaw.push_back(rawFeatureId);
                Splits.emplace_back(std::vector<float>{0.5});
            }
        } else {
            auto splits = BuildSplits(raw.RawFeatures[rawFeatureId], maxBins);
            auto feature = BinarizeFloatFeature(rawColumn, splits);
            pool.Features.emplace_back(std::move(feature));
            BinarizedToRaw.push_back(rawFeatureId);
            Splits.emplace_back(std::move(splits));
        }
    }

    pool.RawFeatureCount = rawFeatureCount;
    pool.BinarizedFeatureCount = pool.Features.size();

    pool.Rows = SetupTestData(pool);

    return pool;
}

TPool TBinarizer::BinarizeTestData(TRawPool&& raw, std::vector<std::vector<float>>& splits) {
    TPool pool;
//    pool.Names = std::move(raw.Names);
//    //we have no target in testing dataset
//    //pool.Target = pool.Target;
//    pool.Size = raw.RawFeatures[0].size();
//
//    pool.Hashes = std::move(raw.Hashes);
//    Splits = std::move(splits);
//
//    auto rawFeatureCount = raw.RawFeatures.size();
//
//    size_t floatFeatureCounter = 0;
//    for (size_t rawFeatureId = 0; rawFeatureId < rawFeatureCount; ++rawFeatureId) {
//        TFeatures binarized;
//        const auto& rawColumn = raw.RawFeatures[rawFeatureId];
//        if (!pool.Hashes[rawFeatureId].empty()) {
//            binarized = BinarizeCatFeature(rawColumn, pool.Hashes[rawFeatureId].size());
//        } else {
//            binarized = BinarizeFloatFeature(rawColumn, Splits[floatFeatureCounter]);
//            floatFeatureCounter += 1;
//        }
//
//        for (auto& column : binarized) {
//            pool.Features.emplace_back(std::move(column));
//            BinarizedToRaw.push_back(rawFeatureId);
//        }
//    }
//
//    pool.RawFeatureCount = raw.RawFeatures.size();
//    pool.BinarizedFeatureCount = pool.Features.size();
//
//    pool.Rows = SetupTestData(pool);

    return pool;
}

/*
TFeatureRow TBinarizer::Binarize(size_t featureId, const std::string& value) const {
    auto& hash = Hashes[featureId];
    auto it = hash.find(value);
    if (it == hash.end()) {
        throw std::logic_error("No such value for categorical feature was seen in the training pool");
    }

    TFeatureRow vector(hash.size());
    vector[it->second] = 1;
    return vector;
}

TFeatureRow TBinarizer::Binarize(size_t featureId, float value) const {
    const auto& splits = Splits[featureId];
    if (splits.empty()) {
        throw std::logic_error("No splits info found for this feature");
    }

    TFeatureRow vector(splits.size());
    for (size_t i = 0; i < splits.size(); ++i) {
        if (value >= splits[i]) {
            vector[i] = 1;
        }
    }
    return vector;
}
*/
TFeature TBinarizer::BinarizeFloatFeature(const TRawFeature& data, const std::vector<float>& splits) {
    size_t length = data.size();
    TFeature binarized(length, 0);

    for (size_t i = 0; i < length; ++i) {
        binarized[i] = uint8_t(std::upper_bound(std::begin(splits), std::end(splits), data[i]) - std::begin(splits));
    }

    return binarized;
}

TFeatures TBinarizer::BinarizeCatFeature(const TRawFeature& data, size_t cats) {
    size_t length = data.size();
    TFeatures binarized(cats, TFeature(length, 0));

    for (size_t i = 0; i < length; ++i) {
        binarized[size_t(data[i])][i] = 1;
    }

    return binarized;
}

TFeatureRows TBinarizer::SetupTestData(const TPool& pool) const {
    TFeatureRows rows(pool.Size, TFeatureRow(pool.BinarizedFeatureCount));
    for (size_t i = 0; i < pool.Size; ++i) {
        for (size_t j = 0; j < pool.BinarizedFeatureCount; ++j) {
            rows[i][j] = pool.Features[j][i];
        }
    }
    return rows;
}


