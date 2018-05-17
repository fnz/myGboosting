#include "predict.h"

#include "algo/binarization.h"
#include "algo/defines.h"
#include "algo/pool.h"
#include "algo/model.h"

#include <iostream>
#include <numeric>
#include <fstream>

void PredictMode::Run(const TPredictConfig& config) {
    std::cout << "Predict" << std::endl;

    std::cout << "Loading Dataset" << config.TestData << std::endl;

    TPool pool;
    TBinarizer binarizer;
    TModel model;

    std::vector<std::vector<float>> splits;
    std::vector<std::unordered_map<std::string, size_t>> hashes;
    model.DeSerialize(config.Model, hashes, splits);

    pool = binarizer.BinarizeTestData(LoadTestingPool(config.TestData, hashes), splits);

    std::cout << "Done" << std::endl;
    std::cout << "Raw features: " << pool.RawFeatureCount << std::endl;
    std::cout << "Binarized features: " << pool.BinarizedFeatureCount << std::endl;
    std::cout << "Size: " << pool.Size << std::endl;

    //TModel model(std::move(binarizer));

    auto predictions = model.Predict(std::move(pool));

    std::cout << "Writing to file: " << config.Output << std::endl;

    std::ofstream out(config.Output);
    for (const auto& val : predictions) {
        //std::cout << val << std::endl;
        out << val << std::endl;
    }

    out.close();
}
