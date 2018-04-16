#include "train.h"

#include "algo/binarization.h"
#include "algo/defines.h"
#include "algo/pool.h"
#include "algo/split.h"
#include "algo/tree.h"

#include <iostream>
#include <numeric>
#include <sstream>

TDecisionTreeNode Terminate(const TPool& pool, TMask& mask) {
    TDecisionTreeNode node;
    node.Leaf = true;
    node.Value = Mean(pool.Target, mask);
    return node;
}

size_t Train(const TPool& pool, TDecisionTree& tree, TMask& mask, size_t depth, size_t maxDepth,
             size_t minCount, bool verbose) {
    if (depth == maxDepth) {
        tree.Nodes.push_back(Terminate(pool, mask));
        if (verbose)
            std::cout << "Depth termination" << std::endl;
        return tree.Nodes.size() - 1;
    }

    auto count = size_t(std::accumulate(mask.begin(), mask.end(), 0));
    if (verbose)
        std::cout << "Count = " << count << std::endl;

    if (count < minCount) {
        tree.Nodes.push_back(Terminate(pool, mask));
        if (verbose)
            std::cout << "Count termination" << std::endl;
        return tree.Nodes.size() - 1;
    }

    TDecisionTreeNode node;
    node.FeatureId = GetOptimalSplit(pool.Features, pool.Target, mask);
    if (verbose)
        std::cout << "Split by feature " << node.FeatureId << std::endl;

    std::vector<size_t> maskIds;
    for (size_t id = 0; id < pool.Size; ++id) {
        if (mask[id] != 0) {
            mask[id] = pool.Features[node.FeatureId][id] >= 0.5;
            maskIds.push_back(id);
        }
    }

    tree.Nodes.push_back(node);
    auto nodeId = tree.Nodes.size() - 1;

    tree.Nodes[nodeId].Right = Train(pool, tree, mask, depth + 1, maxDepth, minCount, verbose);

    for (size_t id : maskIds) {
        mask[id] = pool.Features[node.FeatureId][id] < 0.5;
    }
    tree.Nodes[nodeId].Left = Train(pool, tree, mask, depth + 1, maxDepth, minCount, verbose);

    return nodeId;
}

TFeatureVector ReadLine(const std::string& line, const TBinarizer& binarizer) {
    TFeatureVector vector;

    std::string str;
    std::stringstream stream(line);

    size_t featureId = 0;
    while (std::getline(stream, str, ',')) {
        try {
            float value = std::stof(str);
            for (auto x : binarizer.Binarize(featureId, value)) {
                vector.push_back(x);
            }
        } catch (...) {
            for (auto x : binarizer.Binarize(featureId, str)) {
                vector.push_back(x);
            }
        }
        featureId++;
    }

    return vector;
}

std::vector<float> TrainMode::SingleTreePredictions(const TPool& pool, const TDecisionTree& tree) {
    std::vector<float> predictions(pool.Size, 0);
    for (int k = 0; k < pool.Size; ++k) {
        predictions[k] = tree.PredictPool(pool.Features, k);
    }

    return predictions;
}

std::vector<float> TrainMode::MakePredictions(const TPool& pool, const std::vector<TDecisionTree>& tree_vector) {
    std::vector<float> predictions(pool.Size, 0);
    for (int tree_index = tree_vector.size()-1; tree_index >=0; --tree_index) {
        for (int k = 0; k < pool.Size; ++k) {
            predictions[k] = pool.learning_rate * tree_vector[tree_index].PredictPool(pool.Features, k) + predictions[k];
        }
    }
    return predictions;
}

void TrainMode::Run(const std::string& path, const int iterations, const float learning_rate, const int depth) {
    std::cout << "Train" << std::endl;

    std::cout << "Loading " << path << std::endl;

    TPool pool;
    TBinarizer binarizer;

    pool = binarizer.Binarize(LoadTrainingPool(path));
    pool.learning_rate = learning_rate;

    std::cout << "Done" << std::endl;
    std::cout << "Raw features: " << pool.RawFeatureCount << std::endl;
    std::cout << "Binarized features: " << pool.BinarizedFeatureCount << std::endl;
    std::cout << "Size: " << pool.Size << std::endl;

    std::vector<TDecisionTree> tree_vector;
    TTarget t_backup(pool.Target);
    for (int i = 0; i< iterations; ++i) {
        TDecisionTree tree;
        TMask mask(pool.Size, 1);
        Train(pool, tree, mask, 0, depth, 10, false);
        tree_vector.emplace_back(tree);
        std::vector<float> predictions = SingleTreePredictions(pool, tree);

        //replacing our target by gradient of current step
        for (int row_num=0; row_num < pool.Size; ++row_num)
            pool.Target[row_num] -= pool.learning_rate * predictions[row_num];
    }

    std::vector<float> predictions = MakePredictions(pool, tree_vector);
    //std::vector<float> predictions = SingleTreePredictions(pool, tree_vector[tree_vector.size()-3]);
    float cur_mse = 0.0;
    for (int j = 0; j < pool.Size; ++j) {
        if (t_backup[j] != predictions[j])
            //std::cout << "Actual " << t_backup[j] << " Predicted "<< predictions[j] <<std::endl;
            cur_mse += (t_backup[j] - predictions[j]) * (t_backup[j] - predictions[j]);
    }
    cur_mse /= pool.Size;
    std::cout << "MSE on train: "<< cur_mse << std::endl;
    // Should be 1
    //std::cout << tree.Predict(ReadLine("6.4,2.9,4.3,1.3", binarizer)) << std::endl;



    // Should be 0
    //std::cout << tree.Predict(ReadLine("4.8,3.4,1.6,0.2", binarizer)) << std::endl;

}
