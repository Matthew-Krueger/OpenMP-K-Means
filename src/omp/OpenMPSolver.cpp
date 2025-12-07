//
// Created by Matthew Krueger on 10/13/25.
//

#include "OpenMPSolver.hpp"

#include <algorithm>
#include <ranges>
#include <unordered_set>

#include "../shared/Logging.hpp"

#include "../shared/Utils.hpp"

#include <omp.h>

namespace kmeans {
    OpenMPSolver::OpenMPSolver(Config &config) {
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
        m_NumThreads = config.numThreads;

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


    void OpenMPSolver::run() {
        PROFILE_FUNCTION();

        // set up omp
        omp_set_num_threads(m_NumThreads);

        // so the algorithm is roughly this
        // Calculates the *closest* centroid and class the point as this centroid
        // Then calculate the vector average of all the points
        // the vector average becomes the new
        size_t iteration = 0;
        while (iteration < m_MaxIterations) {
            // test if we have reached convergence or max samples
            PROFILE_SCOPE("Iteration");
            DEBUG_PRINT("OpenMPSolver iteration " << iteration << " of " << m_MaxIterations);

            // in each iteration, we have to class the centroid, then accumulate the centroid to the new average. Generally speaking,
            // while this class -> reduction operation is two separate operations, in this case it may be advantageous to interleave these operations

            // Ergo, we will reserve a new array of current centroids and move the old one to previous
            m_PreviousCentroids = std::move(m_CurrentCentroids);

            // and zero the new one AND
            // since m_CurrentCentroids has counts, we can fill the vector with zero
            //m_CurrentCentroids = std::vector<Point>(m_PreviousCentroids.size(), Point(std::vector<double>(m_DataSet[0].numDimensions(), 0.0), 0));

            // create a global accumulator
            std::vector<Point> globalAccumulators(
                m_PreviousCentroids.size(),
                Point(std::vector<double>(m_DataSet[0].numDimensions(), 0.0), 0)
            );

            // now that we have that, we can now accumulate
            // since OMP does not support std::accumulate, and this pattern is a bit complicated, I am going to
            // use critical in this case
#pragma omp parallel
            {

                // set up the thread local accumulators
                std::vector<Point> localAccumulators(
                    m_PreviousCentroids.size(),
                    Point(std::vector<double>(m_DataSet[0].numDimensions(), 0.0), 0)
                );

                // use nowait so threads can continue when they want
#pragma omp for schedule(static) nowait
                for (size_t m_DataSetPointIndex = 0; m_DataSetPointIndex < m_DataSet.size(); ++m_DataSetPointIndex) {
                    if (auto it = m_DataSet[m_DataSetPointIndex].findClosestPointInVector(m_PreviousCentroids); it != m_PreviousCentroids.end()) {
                        const size_t idx = std::distance(m_PreviousCentroids.begin(), it);

                        // Accumulate into LOCAL vector. No locks needed.
                        localAccumulators[idx] += m_DataSet[m_DataSetPointIndex];
                        localAccumulators[idx].setCount(localAccumulators[idx].getCount() + 1);
                    }
                }

                // lock the global accumulators
                // and then allow each thread to do their thing
                // in this case this particular antipattern is easier than using the custom reductor function
#pragma omp critical
                {
                    for (size_t k = 0; k < globalAccumulators.size(); ++k) {
                        globalAccumulators[k] += localAccumulators[k];
                        globalAccumulators[k].setCount(globalAccumulators[k].getCount() + localAccumulators[k].getCount());
                    }
                }

            } // end omp parallel

            // move global accumulators to m_CurrentCentroids (we do not do this directly, so the accumulation can stay atomic just in case)
            m_CurrentCentroids = std::move(globalAccumulators);

            // normalize the centroid
            // note a prior bug would destroy an empty centroid which has now been corrected
            std::ranges::for_each(m_CurrentCentroids, [this](Point &centroid) {
                if (centroid.getCount() > 0) {
                    centroid /= static_cast<double>(centroid.getCount());
                } else {
                    // Identify which centroid has no count (pointer arithmetic)
                    size_t idx = &centroid - &m_CurrentCentroids[0];

                    // stay where we were
                    centroid = m_PreviousCentroids[idx];
                }
            });

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
