#include "train.h"

#include "algo/binarization.h"
#include "algo/defines.h"
#include "algo/pool.h"
#include "algo/split.h"
#include <algo/odt.h>
#include "algo/tree.h"
#include "algo/model.h"

#include <iostream>
#include <numeric>
#include <sstream>
#include <ctime>

void TrainMode::Run(const TFitConfig& config) {

    std::cout << "Train" << std::endl;

    std::cout << "Loading " << config.TrainData << std::endl;

    TPool pool;
    TBinarizer binarizer;

    pool = binarizer.Binarize(LoadTrainingPool(config.TrainData), config.MaxBins);

    std::cout << "Done" << std::endl;
    std::cout << "Raw features: " << pool.RawFeatureCount << std::endl;
    std::cout << "Binarized features: " << pool.BinarizedFeatureCount << std::endl;
    std::cout << "Size: " << pool.Size << std::endl;

    TModel model(std::move(binarizer));
    model.Fit(std::move(pool), config);

    std::cout << "Saving model to " << config.Model << std::endl;
    model.Serialize(config.Model, pool);
}
