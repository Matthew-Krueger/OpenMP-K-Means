//
// Created by mpiuser on 10/10/25.
//

#include "Point.hpp"

#include <algorithm>
#include <expected>
#include <numeric>
#include <cmath>
#include <ranges>

#include "Instrumentation.hpp"

namespace kmeans {
    double Point::calculateEuclideanDistance(const Point &other) const {
        // don't profile the time is insignificant
        //PROFILE_FUNCTION();

        // Guard against dimension mismatch.
        if (m_Data.size() != other.m_Data.size()) {
            throw std::invalid_argument(
                "Dimensions mismatch. This has " + std::to_string(m_Data.size()) + " dimensions, that has " +
                std::to_string(other.m_Data.size()) + " dimensions.");
        }

        // Lambda function to calculate the squared difference between two doubles.
        auto squaredifference = [](double first, double second) -> double {
            double fms = first - second;
            return fms * fms;
        };

        // Calculate the sum of squared differences using std::transform_reduce.
        // m_Data.begin(), m_Data.end(): The range of elements from the current Point.
        // other.m_Data.begin(): The starting iterator for the other Point's data.
        // 0.0: The initial value for the sum.
        // std::plus<double>(): The binary operation to combine the results of the transform.
        // [squaredifference](auto first, auto second){ ... }: The transform operation, applying squaredifference to corresponding elements.
        double totalSum = std::transform_reduce(
            m_Data.begin(),
            m_Data.end(),
            other.m_Data.begin(),
            0.0,
            std::plus<>(),
            squaredifference
        );

        return std::sqrt(totalSum);
    }

    Point::FlattenedPoints Point::flattenPoints(const std::vector<Point> &points) {
        PROFILE_FUNCTION();

        // Return an error if no points are provided.
        if (points.empty()) {
            throw std::invalid_argument("No points provided");
        }

        // Get the first point's dimensionality, which becomes the expected dimensionality
        const size_t expectedNumberDimensions = points[0].m_Data.size();

        if (expectedNumberDimensions == 0) {
            throw std::invalid_argument("Expected Dimensionality cannot be zero");
        }

        // Check if all points in the vector have the expected number of dimensions.
        bool allHaveRequiredNumberDimensions = std::ranges::all_of(points,
                                                                   [expectedNumberDimensions](const Point &point) {
                                                                       return expectedNumberDimensions == point.m_Data.size();
                                                                   });

        // Return an error if not all points have the same number of dimensions.
        if (!allHaveRequiredNumberDimensions) {
            throw std::invalid_argument("All points must have the same number of dimensions");
        }

        // Construct a result view by joining the data of all points.
        const auto resultView = points | std::ranges::views::join;

        // now use that view to construct a flattened vector
        std::vector<double> flattenedPoints(resultView.begin(), resultView.end());

        return Point::FlattenedPoints{
            expectedNumberDimensions,
            points.size(),
            std::move(flattenedPoints)
        };
    }

    std::vector<Point> Point::unflattenPoints(const FlattenedPoints &flattenedPoints) {
        PROFILE_FUNCTION();

        // Validate that the total number of elements in the flattened vector matches the expected count.
        // totalEntries: The expected total number of elements (numPoints * numDimensionsPerPoint).
        const size_t totalEntries = flattenedPoints.numDimensionsPerPoint * flattenedPoints.numPoints;
        // Check if the actual size of the flattened points vector matches the expected total entries.
        if (flattenedPoints.points.size() != totalEntries) {
            throw std::invalid_argument(
                "Flattened points vector size mismatch. Expected " +
                std::to_string(totalEntries) + " (" +
                std::to_string(flattenedPoints.numPoints) + " points * " +
                std::to_string(flattenedPoints.numDimensionsPerPoint) + " dims) but got " +
                std::to_string(flattenedPoints.points.size()) + "."
            );
        }

        // result: A vector to store the unflattened Point objects.
        std::vector<Point> result;
        // Reserve memory for the expected number of points to avoid reallocations.
        result.reserve(flattenedPoints.numPoints);

        // Iterate through the flattened points data and reconstruct individual Point objects.
        for (size_t currentPointStartingIndex = 0; currentPointStartingIndex < totalEntries;
             currentPointStartingIndex += flattenedPoints.numDimensionsPerPoint) {
            // Extract the data for a single point.
            std::vector<double> pointData(flattenedPoints.points.begin() + static_cast<long>(currentPointStartingIndex),
                                          flattenedPoints.points.begin() + static_cast<long>(currentPointStartingIndex) + static_cast<long>(flattenedPoints.
                                              numDimensionsPerPoint));
            result.emplace_back(pointData);
        }

        return result;
    }

    ClusterLocalAggregateSum ClusterLocalAggregateSum::calculateCentroidLocalSum(const std::vector<Point> &points) {
        PROFILE_FUNCTION();

        // Return an error if no points are provided. This should still run in critical paths.
        if (points.empty()) {
            throw std::invalid_argument("No points provided");
        }

        // Get the first point's dimensionality, which becomes the expected dimensionality
        const size_t expectedNumberDimensions = points[0].getData().size();

        // since this will be an actual critical path, we should wrap this in ifdef ndebug so we can strip it out in release builds
#ifndef NDEBUG


        if (expectedNumberDimensions == 0) {
            throw std::invalid_argument("Expected Dimensionality cannot be zero");
        }

        // Check if all points in the vector have the expected number of dimensions.
        bool allHaveRequiredNumberDimensions = std::ranges::all_of(points,
                                                                   [expectedNumberDimensions](const Point &point) {
                                                                       return expectedNumberDimensions == point.getData().size();
                                                                   });

        // Return an error if not all points have the same number of dimensions.
        if (!allHaveRequiredNumberDimensions) {
            throw std::invalid_argument("All points must have the same number of dimensions");
        }

#endif

        // Sum all elements of each point into one point in a fold left reduction
        // Since fold left uses move semantics
        auto result2 = std::ranges::fold_left(
            points,
            Point(std::vector<double>(expectedNumberDimensions, 0.0)),
            [](Point acc, const Point &p) {
                PROFILE_FUNCTION();

                acc += p;
                return acc;
            }
        );

        // then on each process we can use use operator/ to divide by a scalar.

        return ClusterLocalAggregateSum(std::move(result2),points.size());

    }

    Point& Point::operator+=(const Point &other) {
        PROFILE_FUNCTION();

        // guard against invalid points being added together
        #ifndef NDEBUG
        if (m_Data.size() != other.getData().size() || m_Data.empty()) {
            throw std::invalid_argument("Dimension Mismatch or Invalid dimensions.");
        }
        #endif

        // copy current point, and add each element to it.
        std::ranges::transform(
            m_Data,
            other,
            m_Data.begin(),
            std::plus<>()
        );

        return *this;

    }

    Point& Point::operator/=(const double scalar) {
        PROFILE_FUNCTION();

        // no need to check for compatability in a scalar division
        std::ranges::transform(
            m_Data,
            m_Data.begin(),
            [scalar](const double dimension){ return dimension/scalar ; }
        );

        return *this;

    }

    std::vector<Point>::iterator Point::findClosestPointInVector(std::vector<Point>& other) const {
        PROFILE_FUNCTION();

        // guard against empty iterator
        if (other.empty()) {
            return other.end();
        }

        auto minIter = other.begin();
        // set the min distance to max, that way we can track it.
        double minDist = std::numeric_limits<double>::max();
        bool foundValid = false;

        // iterate through the other vector, and find the closest point to this one.
        // this was one function that should not be functional, for unknown reasons
        for (auto it = other.begin(); it != other.end(); ++it) {
            // if and only if the value is less than minDist, and if and only if it has a value
            if (auto dist = calculateEuclideanDistance(*it); dist < minDist) {
                minDist = dist;
                minIter = it;
                foundValid = true;
            }

        }

        // return what we found, if none, the end iterator
        return foundValid ? minIter : other.end();
    }

    
} // kmeans
