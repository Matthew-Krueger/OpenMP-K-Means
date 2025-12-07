//
// Created by Matthew Krueger on 12/6/25.
//

#ifndef KMEANS_CENTROIDVALIDATOR_HPP
#define KMEANS_CENTROIDVALIDATOR_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>
#include "Point.hpp"


// notes: written by gemini, I am not providing transcripts for this because the way I invoked it does not provide transcripts
namespace kmeans {

    class CentroidValidator {
    public:
        struct ValidationResult {
            /**
             * @brief The sum of squared distances between the matched known and found centroids.
             * Ideally close to 0.0.
             */
            double totalDivergence;

            /**
             * @brief The mapping of indices that produced the minimal divergence.
             * Format: std::pair<KnownCentroidIndex, FoundCentroidIndex>
             */
            std::vector<std::pair<size_t, size_t>> matchedIndices;
        };

        /**
         * @brief Aligns found centroids to known centroids to minimize the Sum of Squared Errors (SSE),
         * then calculates the divergence metric.
         */
        static ValidationResult computeDivergence(const std::vector<Point>& knownCentroids,
                                                  const std::vector<Point>& foundCentroids) {

            if (knownCentroids.empty() || foundCentroids.empty()) {
                return {0.0, {}};
            }

            if (knownCentroids.size() != foundCentroids.size()) {
                throw std::runtime_error("Divergence calculation currently requires equal cluster counts.");
            }

            size_t problemSize = knownCentroids.size();
            Matrix costMatrix = buildCostMatrix(knownCentroids, foundCentroids);

            // Execute Hungarian Algorithm to find optimal assignment based on Squared Euclidean costs
            std::vector<int> optimalAssignment = solveHungarian(costMatrix);

            double totalSquaredDivergence = 0.0;
            std::vector<std::pair<size_t, size_t>> matches;
            matches.reserve(problemSize);

            for (size_t knownIndex = 0; knownIndex < problemSize; ++knownIndex) {
                int assignedFoundIndex = optimalAssignment[knownIndex];

                if (assignedFoundIndex == -1) {
                     throw std::runtime_error("Algorithm failed to assign a match for a centroid.");
                }

                // Re-calculate squared distance for the final metric
                double squaredDistance = squaredEuclideanDistance(knownCentroids[knownIndex], foundCentroids[assignedFoundIndex]);
                totalSquaredDivergence += squaredDistance;

                matches.emplace_back(knownIndex, static_cast<size_t>(assignedFoundIndex));
            }

            return {totalSquaredDivergence, matches};
        }

    private:
        using Matrix = std::vector<std::vector<double>>;

        static double squaredEuclideanDistance(const Point& pointA, const Point& pointB) {
            double sumSquaredDiffs = 0.0;

            // Assuming data is accessible via .getData() returning a vector or similar iterable
            // You may need to adjust this depending on your Point implementation
            const auto& coordinatesA = pointA.getData();
            const auto& coordinatesB = pointB.getData();

            size_t dimensions = coordinatesA.size();

            for(size_t dimensionIndex = 0; dimensionIndex < dimensions; ++dimensionIndex) {
                double diff = coordinatesA[dimensionIndex] - coordinatesB[dimensionIndex];
                sumSquaredDiffs += (diff * diff);
            }

            // No sqrt() here, we want the squared energy
            return sumSquaredDiffs;
        }

        static Matrix buildCostMatrix(const std::vector<Point>& known, const std::vector<Point>& found) {
            size_t size = known.size();
            Matrix matrix(size, std::vector<double>(size));

            for (size_t knownIndex = 0; knownIndex < size; ++knownIndex) {
                for (size_t foundIndex = 0; foundIndex < size; ++foundIndex) {
                    matrix[knownIndex][foundIndex] = squaredEuclideanDistance(known[knownIndex], found[foundIndex]);
                }
            }
            return matrix;
        }

        // --------------------------------------------------------------------------
        // The Hungarian (Munkres) Algorithm Implementation
        // --------------------------------------------------------------------------
        static std::vector<int> solveHungarian(Matrix costMatrix) {
            const size_t matrixSize = costMatrix.size();
            const double infiniteCost = std::numeric_limits<double>::max();

            // FIX: Resize to matrixSize + 1 to accommodate 1-based indexing logic.
            // Index 0 is used as a dummy root for the alternating path.
            std::vector<double> potentialRow(matrixSize + 1, 0.0);
            std::vector<double> potentialColumn(matrixSize + 1, 0.0);

            // stores the Row index assigned to Column [index]. 0 means unassigned.
            std::vector<int> matchForColumn(matrixSize + 1, 0);
            std::vector<int> wayArray(matrixSize + 1, 0);

            // Iterate over each row (1 to N) to find a matching
            for (size_t currentRowIndex = 1; currentRowIndex <= matrixSize; ++currentRowIndex) {
                matchForColumn[0] = static_cast<int>(currentRowIndex);
                size_t currentColumnIndex = 0;

                std::vector<double> minSlackValue(matrixSize + 1, infiniteCost);
                std::vector<bool> isColumnUsed(matrixSize + 1, false);

                // Find augmenting path
                do {
                    isColumnUsed[currentColumnIndex] = true;

                    // Row V that we are currently trying to match from the dummy column
                    size_t assignedRowIndex = static_cast<size_t>(matchForColumn[currentColumnIndex]);

                    double delta = infiniteCost;
                    size_t nextColumnIndex = 0;

                    // Scan columns to calculate slacks and find the best move
                    for (size_t scanColumnIndex = 1; scanColumnIndex <= matrixSize; ++scanColumnIndex) {
                        if (!isColumnUsed[scanColumnIndex]) {

                            // Cost calculation:
                            // costMatrix is 0-indexed, so we subtract 1 from row/col indices.
                            // potentials are 1-indexed, so we access them directly.
                            double reducedCost = costMatrix[assignedRowIndex - 1][scanColumnIndex - 1]
                                                 - potentialRow[assignedRowIndex]
                                                 - potentialColumn[scanColumnIndex];

                            if (reducedCost < minSlackValue[scanColumnIndex]) {
                                minSlackValue[scanColumnIndex] = reducedCost;
                                wayArray[scanColumnIndex] = static_cast<int>(currentColumnIndex);
                            }

                            if (minSlackValue[scanColumnIndex] < delta) {
                                delta = minSlackValue[scanColumnIndex];
                                nextColumnIndex = scanColumnIndex;
                            }
                        }
                    }

                    // Update potentials (Dual variables) to maintain equilibrium
                    for (size_t updateColumnIndex = 0; updateColumnIndex <= matrixSize; ++updateColumnIndex) {
                        if (isColumnUsed[updateColumnIndex]) {
                            potentialRow[matchForColumn[updateColumnIndex]] += delta;
                            potentialColumn[updateColumnIndex] -= delta;
                        } else {
                            minSlackValue[updateColumnIndex] -= delta;
                        }
                    }
                    currentColumnIndex = nextColumnIndex;

                } while (matchForColumn[currentColumnIndex] != 0); // Continue until we find an unassigned column

                // Backtrack update of the matching using the wayArray
                do {
                    size_t previousColumnIndex = static_cast<size_t>(wayArray[currentColumnIndex]);
                    matchForColumn[currentColumnIndex] = matchForColumn[previousColumnIndex];
                    currentColumnIndex = previousColumnIndex;
                } while (currentColumnIndex > 0);
            }

            // Convert 1-based matching to 0-based result vector
            // matchForColumn[Col] = Row. We want Result[Row] = Col.
            std::vector<int> finalAssignment(matrixSize, -1);
            for (size_t columnIndex = 1; columnIndex <= matrixSize; ++columnIndex) {
                if (matchForColumn[columnIndex] > 0) {
                    finalAssignment[static_cast<size_t>(matchForColumn[columnIndex]) - 1] = static_cast<int>(columnIndex) - 1;
                }
            }

            return finalAssignment;
        }
    };
}

#endif //KMEANS_CENTROID_VALIDATOR_HPP
