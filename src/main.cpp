#include "lib/args.hxx"
#include "lib/csv.h"
#include "modes/predict.h"
#include "modes/train.h"

#include <iostream>
#include <fstream>
#include <algo/config.h>
#include <omp.h>

int main(int argc, char** argv) {
    args::ArgumentParser parser("mini Gradient Boosting utility");
    args::Group commands(parser, "commands");
    args::Command fit(commands, "fit", "Builds model based on provided learning dataset and writes it in output file");
    args::Command predict(commands, "predict",
                          "makes predictions with provided model and test dataset, writes it in output file");
    args::Group arguments(parser, "arguments", args::Group::Validators::DontCare, args::Options::Global);
    args::Positional<std::string> input_file(arguments, "input file", "input for train/predict");
    args::ValueFlag<std::string> column_names_file(arguments, "path",
                                                   "file containing dataset column names",  { "column_names" });
    args::ValueFlag<std::string> output_file(arguments, "path", "output for train/predict", { "output" });
    args::ValueFlag<std::string> model_file(arguments, "path", "model for predict", { "model" });
    args::ValueFlag<std::string> target_column(arguments, "column name", "target column name", { "target" });
    args::ValueFlag<size_t> iterations(arguments, "iterations amount", "number of trees in ensemble", { "iterations" }, 100);
    args::ValueFlag<float> learning_rate(arguments, "learning_rate", "trees regularization", { "learning_rate" }, 1.0);
    args::ValueFlag<float> sample_rate(arguments, "sample_rate",
                                       "percentage of rows for each tree (0 to 1.0)", { "sample_rate" }, 0.66);
    args::ValueFlag<size_t> depth(arguments, "tree depth", "decision tree max depth", { "depth" }, 6);
    args::ValueFlag<size_t> max_bins(arguments, "humber of bins", "max number of bins in histogram", { "max_bins" }, 10);
    args::ValueFlag<size_t> min_leaf_count(arguments, "min leaf size",
                                        "min number of samples in leaf node", { "min_leaf_count" }, 10);
    args::ValueFlag<size_t> num_threads(arguments, "number of threads",
                                           "number of threads to use in parallel calculations", { "num_threads" }, 8);
    args::HelpFlag h(arguments, "help", "help", { 'h', "help" });
    //args::PositionalList<std::string> pathsList(arguments, "paths", "files to commit");

    try {
        parser.ParseCLI(argc, argv);

        if (args::get(input_file).empty()) {
            std::cout << "Input file is not set" << std::endl;
            return 1;
        }

        if (args::get(model_file).empty()) {
            std::cout << "Model file is not set" << std::endl;
            return 1;
        }

        auto threads = args::get(num_threads);
        std::cout << "Number of threads: " << threads << std::endl;
        omp_set_num_threads(threads);

        if (fit) {
            TFitConfig config;
            config.TrainData = args::get(input_file);
            config.ColumnNames = args::get(column_names_file);
            config.TargetName = args::get(target_column);
            config.Model = args::get(model_file);
            config.Iterations = args::get(iterations);
            config.Depth = args::get(depth);
            config.MaxBins = args::get(max_bins);
            config.MinLeafSize = args::get(min_leaf_count);
            config.LearningRate = args::get(learning_rate);
            config.SampleRate = args::get(sample_rate);

            TrainMode::Run(config);

        } else if (predict) {
            if (args::get(output_file).empty()) {
                std::cout << "Output file is not set" << std::endl;
                return 1;
            }

            TPredictConfig config;
            config.TestData = args::get(input_file);
            config.Model = args::get(model_file);
            config.Output = args::get(output_file);

            PredictMode::Run(config);
        }
    }
    catch (const args::Help&) {
        std::cout << parser;
    }
    catch (const args::Error& e) {
        std::cerr << e.what() << std::endl << parser;
        return 1;
    }


    return 0;
}