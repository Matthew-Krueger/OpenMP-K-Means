#include <iostream>
#include <random>
#include <string>
#include <boost/program_options.hpp>

#include "shared/DataSet.hpp"
#include "shared/DualOutputStream.hpp"
#include <ranges>

#include "omp/OpenMPSolver.hpp"
#include "serial/SerialSolver.hpp"
#include "shared/CentroidValidator.hpp"
#include "shared/Logging.hpp"
#include "shared/Utils.hpp"
#include "shared/Timer.hpp"

#undef DEBUG_FLAG
#define DEBUG_FLAG true

int main(int argc, char **argv) {
    size_t maxIterations;
    double convergenceThreshold;
    size_t numGeneratedSamples;
    size_t numDimensions;
    size_t numTrueClusters;
    size_t numThreads;
    double clusterSpread;
    long globalSeed;
    bool printHeader;
    std::string filename;
    size_t numTrials;

    try {
        boost::program_options::options_description desc("Allowed options");
        desc.add_options()
                ("help", "produce help message")
                ("max-iterations", boost::program_options::value<size_t>(&maxIterations)->default_value(10000),
                 "Maximum number of iterations to run")
                ("num-samples", boost::program_options::value<size_t>(&numGeneratedSamples)->default_value(1000),
                 "Number of samples in the dataset")
                ("dimensions", boost::program_options::value<size_t>(&numDimensions)->default_value(3),
                 "Number of dimensions. Dimension distribution will be given via global size")
                ("clusters", boost::program_options::value<size_t>(&numTrueClusters)->default_value(3),
                 "Number of clusters - used for both generation and k means parameters")
                ("spread", boost::program_options::value<double>(&clusterSpread)->default_value(3.5),
                 "The standard deviation of the points - i.e. how wide the cluster is")
                ("seed", boost::program_options::value<long>(&globalSeed)->default_value(1234),
                 "Seed for the random number generator - all other sub seeds will be generated from this")
                ("print-header", boost::program_options::bool_switch(&printHeader),
                 "Print the header for the output file")
                ("filename", boost::program_options::value<std::string>(&filename)->default_value("output.csv"),
                 "Filename to write output to")
                ("convergence-threshold",
                 boost::program_options::value<double>(&convergenceThreshold)->default_value(0.0001),
                 "Threshold for convergence.")
                ("trials", boost::program_options::value<size_t>(&numTrials)->default_value(10),
                 "Number of trials to run")
                ("num-threads", boost::program_options::value<size_t>(&numThreads)->default_value(1),
                 "Number of threads to use with openmp");

        boost::program_options::command_line_parser parser{argc, argv};
        parser.options(desc).allow_unregistered().style(
            boost::program_options::command_line_style::default_style |
            boost::program_options::command_line_style::allow_slash_for_short);
        boost::program_options::parsed_options parsed_options = parser.run();

        boost::program_options::variables_map vm;
        boost::program_options::store(parsed_options, vm);
        boost::program_options::notify(vm);

        if (vm.contains("help")) {
            std::cout << desc << '\n';
            return 0;
        }
    } catch (const boost::program_options::error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    // Creating my DualStream
    DualStream ds(std::cout, filename);

    // print the table header
    if (printHeader) {
        ds << "Number Threads," << "Number Samples," << "Number Dimensions," << "Number Clusters," << "Spread," <<
                "Seed," << "Run Time (s)," << "Did Reach Convergence?," << "Iteration Count," <<
                "Max Centroid Difference" << std::endl;
        return 0; // exit after writing the header
    }

    // Create the dataset
    kmeans::DataSet dataSet;
    std::mt19937 generator(globalSeed);
    std::uniform_int_distribution<size_t> subSeedGenerator(1, std::numeric_limits<size_t>::max());

    // create the dataset in a scope so lifetimes can be cleared automatically.
    {
        std::uniform_real_distribution dimensionGenerator(-100000.0, 100000.0);
        // we'll use a large range to make sure we don't get any weird values

        // create our distribution
        auto dimensionRangeView = std::ranges::views::iota(static_cast<size_t>(0), numDimensions)
                                  | std::ranges::views::transform([&dimensionGenerator, &generator](size_t dimension) {
                                      auto a = dimensionGenerator(generator);
                                      auto b = dimensionGenerator(generator);
                                      kmeans::DataSet::Config::ClusterCentroidDimensionDistribution result{
                                          (a < b) ? a : b,
                                          (a >= b) ? a : b
                                      };
                                      return result;
                                  });
        std::vector<kmeans::DataSet::Config::ClusterCentroidDimensionDistribution> dimensionConfig(
            dimensionRangeView.begin(), dimensionRangeView.end());

        kmeans::DataSet::Config datasetConfig{
            dimensionConfig,
            numGeneratedSamples,
            dimensionConfig.size(),
            numTrueClusters,
            clusterSpread,
            subSeedGenerator(generator)
        };

        dataSet = kmeans::DataSet(datasetConfig);
    }

    if constexpr (DEBUG_FLAG) {
        // no main rank check needed at this point since omp only has one thread at this point
        std::cout << "Known Good Centroids:" << std::endl;
        std::ranges::for_each(
            *dataSet.getKnownGoodCentroids(),
            [](auto &point) {
                std::cout << '\t' << point << "\n";
            }
        );
    }

    uint64_t runRandom = subSeedGenerator(generator);

    for (size_t trial = 0; trial < numTrials; ++trial) {
        // now that the dataset is generated, we will split depending on if omp or not
        if (numThreads <= 0) {
            return 0; // this is illogical
        } else if (numThreads == 1) {
            // we will run the serial solver
            // now that we have our dataset, we can actually go to the correct function.
            // runs serial algorithm
            // for the serial algorithm, we'll create a serial solver and go.
            // we don't actually want to time anything other than run, so we'll set it all up first
            kmeans::SerialSolver::Config config(
                maxIterations,
                convergenceThreshold,
                kmeans::DataSet(dataSet.getPoints()),
                runRandom,
                numTrueClusters
            );

            // create the solver
            kmeans::SerialSolver solver(config);

            // and solve
            auto time = timer::time([&solver] {
                solver.run();
            });

            // get the number of threads
            /*auto hungarianTime = timer::time([&solver, &dataSet] {
                if (solver.getCalculatedCentroidsAtCompletion().has_value() && dataSet.getKnownGoodCentroids().has_value()) {
                    return kmeans::CentroidValidator::computeDivergence(solver.getCalculatedCentroidsAtCompletion().value(), dataSet.getKnownGoodCentroids().value());
                } else {
                    throw std::runtime_error("No centroids to compare");
                }
            });

            std::cout << "Hungarian Time: " << hungarianTime.getTimeSecondsDouble() << std::endl;
            std::cout << "Total Squared Euclidian Distance: " << hungarianTime.functionResult.totalDivergence << std::endl;
            */

            // now print our results
            ds << numThreads << ','
                    << numGeneratedSamples << ','
                    << numDimensions << ','
                    << numTrueClusters << ','
                    << clusterSpread << ','
                    << globalSeed << ','
                    << time.getTimeSecondsDouble() << ','
                    << ((maxIterations == solver.getFinalIterationCount()) ? "no" : "yes") << ','
                    << solver.getFinalIterationCount().value_or(0) << ','
                    << kmeans::getMaxCentroidDifference(solver.getCalculatedCentroidsAtCompletion().value(),
                                                        dataSet.getKnownGoodCentroids().value()) << std::endl;

            if constexpr (DEBUG_FLAG) {
                // if we are the main rank, print all centroids
                std::cout << "Calculated Centroids:" << std::endl;
                std::ranges::for_each(
                    *solver.getCalculatedCentroidsAtCompletion(),
                    [](auto &point) {
                        std::cout << '\t' << point << "with a count of " << point.getCount() <<  "\n";
                    }
                );
            }
        } else {
            // we run the omp algorithm
            // we will run the serial solver
            // now that we have our dataset, we can actually go to the correct function.
            // runs serial algorithm
            // for the serial algorithm, we'll create a serial solver and go.
            // we don't actually want to time anything other than run, so we'll set it all up first
            kmeans::OpenMPSolver::Config config(
                maxIterations,
                convergenceThreshold,
                kmeans::DataSet(dataSet.getPoints()),
                runRandom,
                numTrueClusters,
                numThreads
            );

            // create the solver
            kmeans::OpenMPSolver solver(config);

            // and solve
            auto time = timer::time([&solver] {
                solver.run();
            });

            // get the number of threads
            /*auto hungarianTime = timer::time([&solver, &dataSet] {
                if (solver.getCalculatedCentroidsAtCompletion().has_value() && dataSet.getKnownGoodCentroids().has_value()) {
                    return kmeans::CentroidValidator::computeDivergence(solver.getCalculatedCentroidsAtCompletion().value(), dataSet.getKnownGoodCentroids().value());
                } else {
                    throw std::runtime_error("No centroids to compare");
                }
            });

            std::cout << "Hungarian Time: " << hungarianTime.getTimeSecondsDouble() << std::endl;
            std::cout << "Total Squared Euclidian Distance: " << hungarianTime.functionResult.totalDivergence << std::endl;
            */

            // now print our results
            ds << numThreads << ','
                    << numGeneratedSamples << ','
                    << numDimensions << ','
                    << numTrueClusters << ','
                    << clusterSpread << ','
                    << globalSeed << ','
                    << time.getTimeSecondsDouble() << ','
                    << ((maxIterations == solver.getFinalIterationCount()) ? "no" : "yes") << ','
                    << solver.getFinalIterationCount().value_or(0) << ','
                    << kmeans::getMaxCentroidDifference(solver.getCalculatedCentroidsAtCompletion().value(),
                                                        dataSet.getKnownGoodCentroids().value()) << std::endl;

            if constexpr (DEBUG_FLAG) {
                // if we are the main rank, print all centroids
                std::cout << "Calculated Centroids:" << std::endl;
                std::ranges::for_each(
                    *solver.getCalculatedCentroidsAtCompletion(),
                    [](auto &point) {
                        std::cout << '\t' << point << "\n";
                    }
                );
            }
        }
    }

    return
            0;
}
