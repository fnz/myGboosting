#pragma once

#include "defines.h"

#include <vector>

float Mean(const TTarget& data, const TMask& mask);

float Variance(const TTarget& data, const TMask& mask);

std::pair<float, float> GetRange(const TFeature& data, const TMask& mask);

std::vector<float> GetPartition(const TFeature& data, const TMask& mask, size_t parts);

size_t GetOptimalSplit(const TFeatures& features, const TTarget& target, const TMask& mask);