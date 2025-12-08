//
// Created by Matthew Krueger on 10/13/25.
//

#include "SerialSolver.hpp"

#include <algorithm>
#include <ranges>
#include <unordered_set>

#include "../shared/Logging.hpp"

#include "../shared/Utils.hpp"


namespace kmeans {
    SerialSolver::SerialSolver(Config &config) {
        PROFILE_FUNCTION();
        m_DataSet = config.dataSet;

        DEBUG_PRINT("Starting Centroid Count: " << config.startingCentroidCount);
        DEBUG_PRINT("m_DataSet Size: " << m_DataSet.size());

        // guard config size:
        if (config.startingCentroidCount > m_DataSet.size()) {
            throw std::invalid_argument("Cannot select more centroids than data points");
        }

        // generate starting centroids based on DataSet Properties

        // copy config appropriately.
        m_MaxIterations = config.maxIterations;
        m_ConvergenceThreshold = config.convergenceThreshold;
        m_CurrentCentroids = std::vector<Point>();

        size_t dimensionality = m_DataSet[0].numDimensions();
        size_t numCentroids = config.startingCentroidCount;
        size_t seed = config.startingCentroidSeed;

        // reserve space for centroids appropriately. No need to reserve previous as it'll get dumped anyway as soon as we start the run
        m_CurrentCentroids.reserve(config.startingCentroidCount);

        {
            // now, we generate our centroids
            DEBUG_PRINT("Copied Solver Configs");

            // first, create our RNG
            std::mt19937 rng(seed);
            std::uniform_int_distribution<size_t> dist(0, m_DataSet.size() - 1);

            // instead of generating new centroids, we'll just randomly pull existing points BY COPY!
            // So, we'll use an unordered set. There's not a great functional way to do this, and the standard way makes much more sense.
            std::unordered_set<size_t> indices;
            while (indices.size() < numCentroids) {
                size_t index = dist(rng);
                indices.emplace(index);
            }


            std::ranges::transform(indices,
                                   std::back_inserter(m_CurrentCentroids),
                                   [&](const size_t index) { return Point(m_DataSet[index]); }
                                   // explicitly copy the data so we know *FOR SURE* it's unique.
            );
        }
    }


    void SerialSolver::run() {
        PROFILE_FUNCTION();

        // so the algorithm is roughly this
        // Calculates the *closest* centroid and class the point as this centroid
        // Then calculate the vector average of all the points
        // the vector average becomes the new
        size_t iteration = 0;
        while (iteration < m_MaxIterations) {
            // test if we have reached convergence or max samples
            PROFILE_SCOPE("Iteration");
            DEBUG_PRINT("SerialSolver iteration " << iteration << " of " << m_MaxIterations);

            // in each iteration, we have to class the centroid, then accumulate the centroid to the new average. Generally speaking,
            // while this class -> reduction operation is two separate operations, in this case it may be advantageous to interleave these operations

            // Ergo, we will reserve a new array of current centroids and move the old one to previous
            m_PreviousCentroids = std::move(m_CurrentCentroids);

            // and zero the new one AND
            // since m_CurrentCentroids has counts, we can fill the vector with zero
            //m_CurrentCentroids = std::vector<Point>(m_PreviousCentroids.size(), Point(std::vector<double>(m_DataSet[0].numDimensions(), 0.0), 0));

            // now that we have that, we can now accumulate
            // again, this uses move semantics to pass the *same* value back and forth,
            // so the accumulation is a zero cost abstraction that matches the reduction pattern more closely
            // than "just" a for loop
            m_CurrentCentroids = std::accumulate(
                m_DataSet.begin(),
                m_DataSet.end(),
                std::vector<Point>(m_PreviousCentroids.size(),
                                   Point(std::vector<double>(m_DataSet[0].numDimensions(), 0.0), 0)),
                [&](std::vector<Point> acc, Point &point) {
                    PROFILE_SCOPE("Accumulate");
                    auto closestCentroidInPrevious = point.findClosestPointInVector(m_PreviousCentroids);

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

            // transform the m_CurrentCentroids by the scalar
            // so that we have the actual average
            // note a prior bug would destroy an empty centroid which has now been corrected
            size_t largestClusterIndex = 0;
            size_t maxCount = 0;

            for (size_t clusterIndex = 0; clusterIndex < m_CurrentCentroids.size(); ++clusterIndex) {
                if (m_CurrentCentroids[clusterIndex].getCount() > 0) {
                    // Normalize: divide vector sum by count
                    m_CurrentCentroids[clusterIndex] /= static_cast<double>(m_CurrentCentroids[clusterIndex].getCount());

                    // Track the cluster with the most points
                    if (m_CurrentCentroids[clusterIndex].getCount() > maxCount) {
                        maxCount = m_CurrentCentroids[clusterIndex].getCount();
                        largestClusterIndex = clusterIndex;
                    }
                }
            }

            // Pass 2: Teleport "Zombie" (empty) centroids to split the largest cluster
            for (size_t currentCentroidIndex = 0; currentCentroidIndex < m_CurrentCentroids.size(); ++currentCentroidIndex) {
                if (m_CurrentCentroids[currentCentroidIndex].getCount() == 0) {

                    // We found a zombie. We will overwrite it with the largest cluster's centroid
                    // effectively "splitting" the largest cluster into two.

                    // Guard: If all centroids are 0 (start of run or catastrophic failure), do nothing.
                    if (maxCount == 0) continue;

                    Point& source = m_CurrentCentroids[largestClusterIndex];
                    Point& zombie = m_CurrentCentroids[currentCentroidIndex];

                    // Copy the position of the largest cluster
                    zombie = source;

                    // IMPORTANT: We must "Jitter" (offset) them slightly.
                    // If they are identical, they will fight for the exact same points.
                    // By pushing them slightly apart, the K-Means logic will naturally
                    // divide the large cluster between them in the next iteration.

                    const double epsilon = 0.01; // Small offset constant

                    // Modify the first dimension slightly.
                    // (You could modify all dimensions, but one is usually sufficient to break symmetry)
                    if (zombie.numDimensions() > 0) {
                        // Push Zombie +epsilon
                        zombie.getData()[0] += epsilon;

                        // Push Source -epsilon
                        source.getData()[0] -= epsilon;
                    }

                    DEBUG_PRINT("Detected Zombie Centroid [" << i << "]. Teleported to split Cluster [" << largestClusterIndex << "]");
                }
            }

            // now we can check if the centroids have stabilized. If they have, we'll break
            if (areCentroidsConverged(m_PreviousCentroids, m_CurrentCentroids, m_ConvergenceThreshold)) {
                break;
            }

            // now we're done with an iteration.
            if constexpr (DEBUG_FLAG) {
                std::cout << "Iteration " << iteration << std::endl;
                for (auto &centroid: m_CurrentCentroids) {
                    std::cout << centroid << std::endl;
                }
            }

            iteration++;
        }

        m_FinalIterationCount = iteration;
        m_CalculatedCentroidsAtCompletion = m_CurrentCentroids;
        DEBUG_PRINT("Centroids are converged, or terminated due to too many iterations");
    }
} // kmeans
