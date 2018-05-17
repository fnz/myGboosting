#pragma once

#include "defines.h"
#include "pool.h"

#include <vector>

class TBinarizer {
public:
    TPool Binarize(TRawPool&& raw, size_t maxBins);
    TPool BinarizeTestData(TRawPool&& raw, std::vector<std::vector<float>>& splits);
    //TFeatureRow Binarize(size_t featureId, const std::string& value) const;
    //TFeatureRow Binarize(size_t featureId, float value) const;

private:
    TFeature BinarizeFloatFeature(const TRawFeature& data, const std::vector<float>& splits);
    TFeatures BinarizeCatFeature(const TRawFeature& data, size_t cats);
    TFeatureRows SetupTestData(const TPool& pool) const;

public:
    std::vector<size_t> BinarizedToRaw;
    std::vector<std::vector<float>> Splits;
    std::vector<std::unordered_map<std::string, size_t>> Hashes;
};