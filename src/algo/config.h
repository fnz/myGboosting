#pragma once

#include <string>

class TFitConfig {
public:
    std::string TrainData;
    std::string ColumnNames;
    std::string TargetName;
    std::string Model;
    size_t Iterations;
    size_t Depth;
    size_t MaxBins;
    size_t MinLeafSize;
    float LearningRate;
    float SampleRate;
};

class TPredictConfig {
public:
    std::string TestData;
    std::string Model;
    std::string Output;
};