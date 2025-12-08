#include <boost/mpi.hpp>

#include "mpi/MPISolver.hpp"
#include "serial/OpenMPSolver.hpp"
#include "shared/DataSet.hpp"
#include "shared/Logging.hpp"
#include "shared/Point.hpp"
#include "shared/Instrumentation.hpp"
#include <boost/program_options.hpp>
#include <ranges>
#include "shared/Timer.hpp"
#include "shared/DualOutputStream.hpp"
#include "shared/Utils.hpp"

void serial_profiler() {

}

void mpi_profiler() {

}

int main(int argc, char **argv) {
    DEBUG_PRINT("Creating MPI Environment");
    boost::mpi::environment mpiEnvironment(argc, argv);
    boost::mpi::communicator worldCommunicator;

    size_t maxIterations;
    double convergenceThreshold;
    size_t numGeneratedSamples;
    size_t numDimensions;
    size_t numTrueClusters;
    double clusterSpread;
    long globalSeed;
    bool printHeader;
    std::string filename;
    size_t numTrials;

    try {
        boost::program_options::options_description desc("Allowed options");
        desc.add_options()
                ("help", "produce help message")
                ("max-iterations", boost::program_options::value<size_t>(&maxIterations)->default_value(10000), "Maximum number of iterations to run")
                ("num-samples", boost::program_options::value<size_t>(&numGeneratedSamples)->default_value(1000), "Number of samples in the dataset")
                ("dimensions", boost::program_options::value<size_t>(&numDimensions)->default_value(3), "Number of dimensions. Dimension distribution will be given via global size")
                ("clusters", boost::program_options::value<size_t>(&numTrueClusters)->default_value(3), "Number of clusters - used for both generation and k means parameters")
                ("spread", boost::program_options::value<double>(&clusterSpread)->default_value(3.5), "The standard deviation of the points - i.e. how wide the cluster is")
                ("seed", boost::program_options::value<long>(&globalSeed)->default_value(1234), "Seed for the random number generator - all other sub seeds will be generated from this")
                ("print-header", boost::program_options::bool_switch(&printHeader), "Print the header for the output file")
                ("filename", boost::program_options::value<std::string>(&filename)->default_value("output.csv"), "Filename to write output to")
                ("convergence-threshold", boost::program_options::value<double>(&convergenceThreshold)->default_value(0.0001), "Threshold for convergence.")
                ("trials", boost::program_options::value<size_t>(&numTrials)->default_value(10), "Number of trials to run");

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
    if (printHeader && worldCommunicator.rank() == 0) {
        ds << "Number Processes," << "Number Samples," << "Number Dimensions," << "Number Clusters," << "Spread," << "Seed," << "Run Time (s)," << "Did Reach Convergence?," << "Iteration Count," << "Max Centroid Difference" << std::endl;
        return 0; // exit after writing the header
    }

    auto writer = new instrumentation::MPIWriter(instrumentation::MPIWriter::Config{
        "log.json",
        0,
        5020,
        0
    });

    PROFILE_BEGIN_SESSION(std::unique_ptr<instrumentation::MPIWriter>(writer));

    // we'll create our dataset no matter what
    // in a child scope so we can dump all associated data quickly
    kmeans::DataSet dataSet;
    std::mt19937 generator(globalSeed);
    std::uniform_int_distribution<size_t> subSeedGenerator(1, std::numeric_limits<size_t>::max());
    std::uniform_real_distribution dimensionGenerator(-100000.0, 100000.0); // we'll use a large range to make sure we don't get any weird values

    {

        // create our distribution
        auto dimensionRangeView = std::ranges::views::iota(static_cast<size_t>(0), numDimensions)
            | std::ranges::views::transform([&dimensionGenerator, &generator](size_t dimension) {
                auto a = dimensionGenerator(generator);
                auto b = dimensionGenerator(generator);
                kmeans::DataSet::Config::ClusterCentroidDimensionDistribution result{
                    (a<b) ? a : b,
                    (a>=b) ? a : b
                };
                return result;
        });
        std::vector<kmeans::DataSet::Config::ClusterCentroidDimensionDistribution> dimensionConfig(dimensionRangeView.begin(), dimensionRangeView.end());

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
        if (worldCommunicator.rank() == 0) {
            // if we are the main rank, print all centroids
            std::cout << "Known Good Centroids:" << std::endl;
            std::ranges::for_each(
                *dataSet.getKnownGoodCentroids(),
                [](auto &point) {
                    std::cout << '\t' << point << "\n";
                }
            );
        }
    }

    uint64_t runRandom = subSeedGenerator(generator);

    for (size_t trial = 0; trial < numTrials; ++trial) {
        // now that we have our dataset, we can actually go to the correct function.
        // note, we are implicitly going to be calling our serial code when world size is one
        if (worldCommunicator.size() == 1) {
            // runs serial algorithm
            // for the serial algorithm, we'll create a serial solver and go.
            // we don't actually want to time anything other than run, so we'll set it all up first
            kmeans::OpenMPSolver::Config config(
                maxIterations,
                convergenceThreshold,
                kmeans::DataSet(dataSet.getPoints()),
                runRandom,
                numTrueClusters
            );

            // create the solver
            kmeans::OpenMPSolver solver(config);

            // and solve
            auto time = timer::time([&solver] {
                solver.run();
            });

            // now print our results
            if (worldCommunicator.rank() == 0) {
                ds  << worldCommunicator.size() << ','
                    << numGeneratedSamples << ','
                    << numDimensions << ','
                    << numTrueClusters << ','
                    << clusterSpread << ','
                    << globalSeed << ','
                    << time.getTimeSecondsDouble() << ','
                    << ((maxIterations == solver.getFinalIterationCount()) ? "no" : "yes") << ','
                    << solver.getFinalIterationCount().value_or(0) << ','
                    << kmeans::getMaxCentroidDifference(solver.getCalculatedCentroidsAtCompletion().value(), dataSet.getKnownGoodCentroids().value()) << std::endl;;
            }

            if constexpr (DEBUG_FLAG) {
                if (worldCommunicator.rank() == 0) {
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

        } else {
            // runs MPI Algorithm.
            kmeans::MPISolver::Config config(
                maxIterations,
                convergenceThreshold,
                kmeans::DataSet(dataSet.getPoints()),
                runRandom,
                numTrueClusters,
                0,
                2550
            );
            kmeans::MPISolver solver(std::move(config), worldCommunicator);

            auto time = timer::time([&solver] {
                solver.run();
            });

            if (worldCommunicator.rank() == 0) {
                ds  << worldCommunicator.size() << ','
                    << numGeneratedSamples << ','
                    << numDimensions << ','
                    << numTrueClusters << ','
                    << clusterSpread << ','
                    << globalSeed << ','
                    << time.getTimeSecondsDouble() << ','
                    << ((maxIterations == solver.getFinalIterationCount()) ? "no" : "yes") << ','
                    << solver.getFinalIterationCount().value_or(0) << ','
                    << kmeans::getMaxCentroidDifference(solver.getCalculatedCentroidsAtCompletion().value(), dataSet.getKnownGoodCentroids().value()) << std::endl;
            }

            if constexpr (DEBUG_FLAG) {
                if (worldCommunicator.rank() == 0) {
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
    }


    PROFILE_END_SESSION();

}

//     auto writer = new instrumentation::MPIWriter(instrumentation::MPIWriter::Config{
//         "log.json",
//         0,
//         5020,
//         0
//     });
//
//     PROFILE_BEGIN_SESSION(std::unique_ptr<instrumentation::MPIWriter>(writer));
//
//     {
//         PROFILE_SCOPE("Serial Run");
//         DEBUG_PRINT("Finished creating MPI Environment");
//         DEBUG_PRINT("Creating Dataset ");
//
//         kmeans::DataSet::Config datasetConfig{
//             {{0,10}, {10,20}, {20,30}},
//             10000,
//             3,
//             20,
//             3.5,
//             1
//         };
//
//
//         kmeans::DataSet dataSet = kmeans::DataSet();
//         std::optional<std::vector<kmeans::Point>> knownGoodCentroids;
//         if (worldCommunicator.rank() == 0) {
//             dataSet = kmeans::DataSet(datasetConfig);
//             knownGoodCentroids = dataSet.getKnownGoodCentroids();
//         }
//
//         DEBUG_PRINT("Finished creating Dataset");
//
//         DEBUG_PRINT("Printing known good centroids");
//         if (dataSet.getKnownGoodCentroids().has_value()) {
//             for (auto&point : dataSet.getKnownGoodCentroids().value()) {
//                 std::cout << point << std::endl;
//             }
//         }
//
//         /*DEBUG_PRINT("Creating solver");
//         kmeans::SerialSolver::Config solverConfig(
//                 1000,
//                 0.0001,
//                 std::move(dataSet),
//                 1234,
//                 35
//             );
//
//         DEBUG_PRINT("Created Solver Config");
//         kmeans::SerialSolver solver(solverConfig);
//
//         solver.run();*/
//
//         DEBUG_PRINT("Parallel Solver");
//
//         kmeans::MPISolver::Config solverConfig({
//             10000,
//             0.0001,
//             std::move(dataSet),
//             1234,
//             20,
//             0,
//             1
//         });
//
//         DEBUG_PRINT("Created Solver Config");
//         DEBUG_PRINT("Rank " << worldCommunicator.rank() << ". Initalize Solver");
//         kmeans::MPISolver mpiSolver(std::move(solverConfig), worldCommunicator);
//
//         mpiSolver.run();
//
//         //std::cout << "I am rank " << worldCommunicator.rank() << " of a world size " << worldCommunicator.size() << std::endl;
//         if (worldCommunicator.rank() == 0) {
//             std::cout << "Known good centroids:" << std::endl;
//             if (knownGoodCentroids.has_value()) {
//                 for (auto&point : knownGoodCentroids.value()) {
//                     std::cout << "\tKnown: " << point << std::endl;
//                 }
//             }
//             std::cout << "Final Centroids:" << std::endl;
//             if (mpiSolver.getCalculatedCentroidsAtCompletion().has_value()) {
//                 for (auto&point : mpiSolver.getCalculatedCentroidsAtCompletion().value()) {
//                     std::cout << "\tCalculated:" << point << std::endl;
//                 }
//             }
//             std::cout << "Iterations " << mpiSolver.getFinalIterationCount().value_or(0) << std::endl;
//         }
//
//         PROFILE_END_SESSION();
//
//     }
//}

