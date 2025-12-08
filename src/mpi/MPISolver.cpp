//
// Created by Matthew Krueger on 10/15/25.
//

#include "MPISolver.hpp"
#include <algorithm>
#include <unordered_set>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <ranges>
#include <boost/serialization/vector.hpp>
#include "../shared/Logging.hpp"
#include <boost/mpi/operations.hpp>

#include "../shared/Utils.hpp"

namespace kmeans {

    struct AddPointsFuctor {

        std::vector<Point> operator()(const std::vector<Point>& a, const std::vector<Point>& b) const {
            std::vector<Point> finalResult;
            finalResult.reserve(a.size());
            std::ranges::transform(a, b, std::back_inserter(finalResult), [&](const Point& left, const Point& right) {
                Point result = left + right;
                result.setCount(left.getCount() + right.getCount());
                return result;
            });
            return finalResult;

        }

    };

    MPISolver::MPISolver(Config &&config, boost::mpi::communicator &communicator) : m_Communicator(communicator) {
        PROFILE_FUNCTION();

        DEBUG_PRINT("Rank " << m_Communicator.rank() << ". Creating Solver from config");
        m_CurrentCentroids = std::vector<Point>();
        m_CurrentCentroids.reserve(config.startingCentroidCount); // reserve the right number of centroids
        m_PreviousCentroids = std::vector<Point>(); // need not reserve since it will get dumped anyway
        m_MaxIterations = config.maxIterations;
        m_ConvergenceThreshold = config.convergenceThreshold;
        m_MainRank = config.mainRank;
        m_WorkingTag = config.workingTag;

        DEBUG_PRINT("Rank " << m_Communicator.rank() << ". Finished creating solver from config. Configuring");

        // before we distribute the dataset, we'll get the random points to make our centroids on the main rank only
        if (m_Communicator.rank() == m_MainRank) {
            DEBUG_PRINT("Rank " << m_Communicator.rank() << ". Creating initial centroids from dataset");
            // first, create our RNG
            std::mt19937 rng(config.startingCentroidSeed);
            std::uniform_int_distribution<size_t> dist(0, config.dataSet.size() - 1);

            // instead of generating new centroids, we'll just randomly pull existing points BY COPY!
            // So, we'll use an unordered set. There's not a great functional way to do this, and the standard way makes much more sense.
            std::unordered_set<size_t> indices;
            while (indices.size() < config.startingCentroidCount) {
                size_t index = dist(rng);
                indices.emplace(index);
            }

            // then copy by value
            std::ranges::transform(indices,
                                   std::back_inserter(m_CurrentCentroids),
                                   [&](const size_t index) { return Point(config.dataSet[index]); }
                                   // explicitly copy the data so we know *FOR SURE* it's unique.
            );
        }

        // now that we have our centroids, we can distribute our centroids.
        // we are sending an rValue so that we don't copy dataset
        initialDistributeDataSet(std::move(config.dataSet));
        initialDistributeCentroids();

        // now, every rank should have its own unique dataset, and we should be good
    }

    void MPISolver::run() {
        PROFILE_FUNCTION();

        DEBUG_PRINT("Rank " << m_Communicator.rank() << " - Starting Centroid Count: " << m_CurrentCentroids.size());
        DEBUG_PRINT("Rank " << m_Communicator.rank() << " - m_LocalDataSet Size: " << m_LocalDataSet.size());


        size_t iteration = 0;
        while (iteration < m_MaxIterations) { // test if we have reached convergence or max samples

            // in each iteration, we have to class the centroid, then accumulate the centroid to the new average.
            // then, we sync the centroids (accumulate globally), and then we repeat

            // like serial code, it is most efficent to class and accumulate in one action

            // first step is to move the current centroids to the previous
            m_PreviousCentroids = std::move(m_CurrentCentroids);

            // and zero the new one AND
            // since m_CurrentCentroids has counts, we can fill the vector with zero
            //m_CurrentCentroids = std::vector<Point>(m_PreviousCentroids.size(), Point(std::vector<double>(m_DataSet[0].numDimensions(), 0.0), 0));

            // now that we have that, we can now accumulate
            // again, this uses move semantics to pass the *same* value back and forth,
            // so the accumulation is a zero cost abstraction that matches the reduction pattern more closely
            // than "just" a for loop
            // echo for stuff
            DEBUG_PRINT("BEFORE ACCUMULATE\n" <<
                        "Rank " << m_Communicator.rank() << " has " << m_CurrentCentroids.size() << " centroids"
                <<"\n\t has " << m_LocalDataSet.size() << " points"
                <<"\n\t has " << m_PreviousCentroids.size() << " previous centroids");
            m_CurrentCentroids = std::accumulate(
                m_LocalDataSet.begin(),
                m_LocalDataSet.end(),
                std::vector<Point>(m_PreviousCentroids.size(),
                                   Point(std::vector<double>(m_PreviousCentroids[0].numDimensions(), 0.0), 0)),
                [&](std::vector<Point> acc, Point &point) {
                    PROFILE_SCOPE("Accumulate");
                    auto closestCentroidInPrevious = point.findClosestPointInVector(m_PreviousCentroids); // class to previous

                    if (closestCentroidInPrevious != m_PreviousCentroids.end()) {
                        size_t centroidIndex = std::distance(m_PreviousCentroids.begin(), closestCentroidInPrevious);
                        acc[centroidIndex] += point;
                        acc[centroidIndex].setCount(acc[centroidIndex].getCount() + 1);
                        return acc;
                    } else {
                        throw std::runtime_error("Centroid not found in previous centroids");
                    }
                }
            );

            // now, our m_CurrentCentroids contains our *LOCAL* sum.
            // we need to sync them through an allreduce
            // and then we can divide them by the scalar.
            // echo for stuff
            DEBUG_PRINT("BEFORE GLOBAL REDUCTION" << std::endl <<
                        "Rank " << m_Communicator.rank() << " has " << m_CurrentCentroids.size() << " centroids"
                <<"\n\t has " << m_LocalDataSet.size() << " points"
                <<"\n\t has " << m_PreviousCentroids.size() << " previous centroids");
            globalReduceCentroids();

            // echo for stuff
            DEBUG_PRINT("BEFORE SCALAR\n" <<
                        "Rank " << m_Communicator.rank() << " has " << m_CurrentCentroids.size() << " centroids"
                <<"\n\t has " << m_LocalDataSet.size() << " points"
                <<"\n\t has " << m_PreviousCentroids.size() << " previous centroids");

            std::ranges::for_each(m_CurrentCentroids, [](Point &centroid) {
                if (centroid.getCount() > 0) {
                    centroid /= static_cast<double>(centroid.getCount());
                    centroid.setCount(1); // we need to set the count back to one
                }
                // If getCount() is 0, the centroid sum is already {0,0,...}, which is correct for an empty cluster.
            });

            // echo for stuff
            DEBUG_PRINT("BEFORE CONVERGE\n" <<
                        "Rank " << m_Communicator.rank() << " has " << m_CurrentCentroids.size() << " centroids"
                <<"\n\t has " << m_LocalDataSet.size() << " points"
                <<"\n\t has " << m_PreviousCentroids.size() << " previous centroids");

            // so, now that we have applied the centroids, since all ranks should be identical,
            // so now we can use the heuristic to check for early stopping on each rank. There's no real reason to do this on one thread and broadcast as we'll be waiting anyway
            // now we can check if the centroids have stabilized. If they have, we'll break
            if (areCentroidsConverged(m_PreviousCentroids, m_CurrentCentroids, m_ConvergenceThreshold)) {
                break;
            }

            // now we're done with an iteration.
            if constexpr (DEBUG_FLAG) {
                std::cout << "Rank " << m_Communicator.rank() << "Iteration " << iteration << std::endl;
            }

            // echo for stuff
            DEBUG_PRINT("END OF ITERATION\n" <<
                        "Rank " << m_Communicator.rank() << " has " << m_CurrentCentroids.size() << " centroids"
                <<"\n\t has " << m_LocalDataSet.size() << " points"
                <<"\n\t has " << m_PreviousCentroids.size() << " previous centroids");

            ++iteration;

        }

        m_FinalIterationCount = iteration;
        m_CalculatedCentroidsAtCompletion = m_CurrentCentroids;

    }

    void MPISolver::initialDistributeDataSet(DataSet &&dataSet) {
        PROFILE_FUNCTION();
        // clear our local dataset so we can later insert
        m_LocalDataSet = DataSet();

        if (m_Communicator.rank() == m_MainRank) {
            PROFILE_SCOPE("Main Rank");

            // we are main rank and thus hold the dataset
            // calculate partition keys
            int dataSetSize = static_cast<int>(dataSet.size());
            int numPartitions = m_Communicator.size();
            int dataSetPartitionAmount = dataSetSize / numPartitions;
            int dataSetRemainder = dataSetSize % numPartitions;

            int currentCentroidsSize = static_cast<int>(m_CurrentCentroids.size());

            // I may not need these so commented for now, but may use them
            // int currentCentroidsPartitionKey = currentCentroidsSize / numPartitions;
            // int currentCentroidsRemainder = currentCentroidsSize % numPartitions;

            // we need to calculate the displacements and sizes
            std::vector<int> displacements;
            std::vector<int> sizes;
            displacements.reserve(numPartitions);
            sizes.reserve(numPartitions);

            DEBUG_PRINT("Rank " << m_Communicator.rank() << ". Num Partitions: " << numPartitions);
            DEBUG_PRINT("Rank " << m_Communicator.rank() << ". Data Set Partition Amount: " << dataSetPartitionAmount);
            DEBUG_PRINT("Rank " << m_Communicator.rank() << ". Data Set Remainder: " << dataSetRemainder);
            DEBUG_PRINT("Rank " << m_Communicator.rank() << ". Current Centroids Size: " << currentCentroidsSize);
            std::ranges::for_each(std::ranges::views::iota(0, numPartitions), [&](int currentPartition) {
                const unsigned char remainderAmount = (currentPartition < dataSetRemainder) ? 1 : 0;
                if (currentPartition > 0) {
                    // there is a previous partition
                    displacements.emplace_back(displacements.back() + sizes.back());
                    // the current displacement is the previous displacement plus the
                } else {
                    displacements.emplace_back(0); // the first displacement is 0
                }
                sizes.emplace_back(dataSetPartitionAmount + remainderAmount); // sizes are easy.
                // [a,b,c,d,e,f,g,h,i,j]
                // [0,4,7]
                // [4,3,3]
            });

            // create our local buffers
            std::vector<Point> rawLocalDataSet;
            int rawLocalDataSetSize = 0;

            // we can scatter the sizes
            {
                PROFILE_SCOPE("Scattering sizes");

                DEBUG_PRINT("Rank " << m_Communicator.rank() << ". Scattering sizes" << std::endl
                    << "\trawLocalDataSet " << rawLocalDataSet.size() << std::endl
                    << "\trawLocalDataSetSize " << rawLocalDataSetSize << std::endl
                    << "\tdisplacements " << displacements.size() << std::endl
                    << "\tsizes " << sizes.size() << std::endl
                );
                boost::mpi::scatter(m_Communicator, sizes, rawLocalDataSetSize, m_MainRank);
            }

            // now, allocate space in rawLocalDataSet
            rawLocalDataSet.resize(rawLocalDataSetSize);

            DEBUG_PRINT("Rank " << m_Communicator.rank() << " rawLocalDataSet size " << rawLocalDataSet.size());

            // now that we've sanity checked, it's time to scatter the points
            {
                PROFILE_SCOPE("Scattering dataset");
                boost::mpi::scatterv(
                    m_Communicator,
                    dataSet.getPoints(),
                    sizes,
                    displacements,
                    rawLocalDataSet.data(),
                    rawLocalDataSetSize,
                    m_MainRank
                );
            }

            // now that we've scattered, we can move the data into our local dataset
            m_LocalDataSet = DataSet(std::move(rawLocalDataSet));

            // next, we need to scatter our initial centroids

        } else {
            PROFILE_SCOPE("Worker Rank");

            if (dataSet.size() != 0) {
                DEBUG_PRINT("Dataset size: " << dataSet.size() << " is illogical. Only main rank should have data");
            }
            // create our local buffers
            std::vector<Point> rawLocalDataSet;
            int rawLocalDataSetSize = 0;

            // we can scatter the sizes
            {
                PROFILE_SCOPE("Scattering sizes");
                DEBUG_PRINT("Rank " << m_Communicator.rank() << ". Scattering sizes" << std::endl
                    << "\trawLocalDataSet " << rawLocalDataSet.size() << std::endl
                    << "\trawLocalDataSetSize " << rawLocalDataSetSize << std::endl
                    //<< "\tdisplacements " << displacements.size() << std::endl
                    //<< "\tsizes " << sizes.size() << std::endl
                );
                // scatter the sizes
                boost::mpi::scatter(m_Communicator, std::vector<int>(), rawLocalDataSetSize, m_MainRank);
            }

            // now, allocate space in rawLocalDataSet
            rawLocalDataSet.resize(rawLocalDataSetSize);
            DEBUG_PRINT("Rank " << m_Communicator.rank() << " rawLocalDataSet size " << rawLocalDataSet.size());


            // now that we've sanity checked, it's time to scatter the dataset
            {
                PROFILE_SCOPE("Scattering dataset");
                boost::mpi::scatterv(
                    m_Communicator,
                    std::vector<Point>(),
                    std::vector<int>(),
                    std::vector<int>(),
                    rawLocalDataSet.data(),
                    rawLocalDataSetSize,
                    m_MainRank
                );
            }

            // now that we've scattered, we can move the data into our local dataset
            m_LocalDataSet = DataSet(std::move(rawLocalDataSet));
        }
    }

    void MPISolver::initialDistributeCentroids() {
        PROFILE_FUNCTION();

        boost::mpi::broadcast(m_Communicator, m_CurrentCentroids, m_MainRank);

        if constexpr (DEBUG_FLAG) {
            if (m_Communicator.rank() == m_MainRank) {
                std::cout << "Main Rank Initial Centroids after broadcast (Rank " << m_Communicator.rank() << "): " << m_CurrentCentroids.size() << std::endl;
            } else {
                std::cout << "Worker Rank " << m_Communicator.rank() << " Initial Centroids after broadcast (Rank " << m_Communicator.rank() << "): " << m_CurrentCentroids.size() << std::endl;
            }
        }

    }

    void MPISolver::globalReduceCentroids() {
        PROFILE_FUNCTION();

        // since we are doing a reduction we need to flatten it into MPI primitives
        std::vector<Point> newCentroids;
        newCentroids.reserve(m_CurrentCentroids.size());
        boost::mpi::all_reduce(m_Communicator, m_CurrentCentroids, newCentroids, AddPointsFuctor());
        m_CurrentCentroids = std::move(newCentroids);

    }

    void MPISolver::globalGatherCentroids(const std::vector<Point> &localCentroids) {
        PROFILE_FUNCTION();

        int size = static_cast<int>(localCentroids.size());
        std::vector<int> sizes;
        boost::mpi::all_gather(m_Communicator, size, sizes);

        // gather all local centroids.
        boost::mpi::all_gatherv(m_Communicator, localCentroids, m_CurrentCentroids, sizes);

    }


    void MPISolver::applyScalarToCentroids(std::vector<Point> &centroids) {
        PROFILE_FUNCTION();

        std::ranges::for_each(centroids, [](Point &centroid) {
            if (centroid.getCount() > 0) {
                centroid /= static_cast<double>(centroid.getCount());
            }
            // If getCount() is 0, the centroid sum is already {0,0,...}, which is correct for an empty cluster.
        });

    }


}
