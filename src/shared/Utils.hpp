//
// Created by Matthew Krueger on 10/13/25.
//

#ifndef KMEANS_MPI_UTILS_HPP
#define KMEANS_MPI_UTILS_HPP
#include <algorithm>
#include <cmath>
#include <ranges>
#include <vector>
#include <optional>
#include <unordered_set>

#include "Point.hpp"

namespace kmeans {

    /* we actually want to compare by euclidian distance not direct element comparison, so I'm just removing this I did not realize this unfortunately
    inline bool areVectorsPracticallyTheSame(const std::vector<double>& lhs, const std::vector<double>& rhs, std::optional<double> epsilon = std::nullopt) {

        // guard against unequal vectors
        // since we don't care the *reason* it's invalid, we can just return false
        if (lhs.size()!=rhs.size()) {
            return false;
        }

        // set epsilon to numeric limits episilon if not set
        if (!epsilon.has_value()) {
            epsilon = std::numeric_limits<double>::epsilon();
        }

        // zip up the two vectors to iterate through them together.
        auto vectorView = std::ranges::views::zip(lhs, rhs);

        // accumulate the truth of the similarity, according to the formula in the lambda
        return std::ranges::all_of(
            vectorView.begin(),
            vectorView.end(),
            [epsilon](auto && pairElements) {
                // subtract the absolute values of each element, and they are "equal" if they are smaller than epsilon
                return (std::fabs(std::get<0>(pairElements)) - std::fabs(std::get<1>(pairElements))) < *epsilon;
            }
        );

    } */

    inline bool areCentroidsConverged(const std::vector<Point>& lhs, const std::vector<Point>& rhs, std::optional<double> epsilon = std::nullopt) {

        auto centroidCombinedView = std::ranges::views::zip(lhs, rhs);
        return std::ranges::all_of(
            centroidCombinedView.begin(),
            centroidCombinedView.end(),
            [epsilon](auto && pair) {
                auto result = std::get<0>(pair).calculateEuclideanDistance(std::get<1>(pair));
                return std::fabs(result) < *epsilon;
            });

    }

    inline double getMaxCentroidDifference(const std::vector<Point>& lhs, const std::vector<Point>& rhs) {

        return 0.0;

    }

        /* what I wrote is not correct
        auto centroidCombinedView = std::ranges::views::zip(lhs, rhs);
        auto distances = std::ranges::transform_view(
            centroidCombinedView,
            [](auto && pair) {
                if (auto distance = std::get<0>(pair).calculateEuclideanDistance(std::get<1>(pair)); distance.has_value()) {
                    return *distance;
                }else {
                    throw std::runtime_error("Distance was not calculated");
                }
            }
        );

        return std::ranges::max(distances);*/

    //}

}

#endif //KMEANS_MPI_UTILS_HPP