//
// Created by Matthew Krueger on 10/10/25.
//

#include "DataSet.hpp"

#include <algorithm>
#include <ranges>
#include <boost/random/normal_distribution.hpp>
#include <memory>

#include "Instrumentation.hpp"

namespace kmeans {

    // Choose treachery and evil auto return,
    // It's more FUN!
    // Uncle Iroh in The Ember Island Players (probably)
    auto DataSet::generateCluster(const Point &clusterCenter, size_t numberPoints, double clusterSpread,
                                  std::shared_ptr<std::mt19937> rng) {
        PROFILE_FUNCTION();

        // reserve the vector
        size_t clusterNumberDimensions = clusterCenter.getData().size();
        auto distributions = std::make_shared<std::vector<boost::normal_distribution<double>>>();
        distributions->reserve(clusterNumberDimensions);

        // use std::ranges::transform to populate distributions with gauntness distributions *BASED ON* the cluster center
        // back inserter is used to make sure the distributions iterator stays valid
        std::ranges::transform(clusterCenter,
                               std::back_inserter(*distributions),
                               [clusterSpread](double dimension) {
                                   PROFILE_FUNCTION();
                                   return boost::normal_distribution<double>(dimension, clusterSpread);
                               }
        );

        // we will now generate points with each index in the vector using its own corresponding boost distribution.
        auto generatedPointsView = std::ranges::views::iota((size_t)0, numberPoints)
                           | std::ranges::views::transform([distributions, rng](int) { return generateSinglePoint(distributions, rng); });

        return generatedPointsView;

    }

    DataSet::DataSet(const Config& config) {
        PROFILE_FUNCTION();

        // First, we should make sure the config parameter is valid
        if (config.numTrueClusters>config.numTotalSamples) {
            throw std::invalid_argument("Number of clusters cannot be greater than number of total samples");
        }

        if (config.clusterDimensionDistributions.size()!=config.numDimensions) {
            throw std::invalid_argument("Dimension Distributions does not contain expected number of dimensions");
        }

        // reserve the right number of samples
        m_Points.reserve(config.numTotalSamples);

        // create our random
        auto rng = std::make_shared<std::mt19937>(config.seed);

        // scope the generation of our known good centroids
        // so we can dump the distribution generators ASAP
        {
            PROFILE_SCOPE("Create distributions of the clusters themselves");
            // create our distributions (we are using linear distributions for now
            // This is on a PER DIMENSION BASIS.
            // ESSENTIALLY, WE NEED TO HAVE A VECTOR OF DISTRIBUTIONS SO THAT WE CAN THEN MAKE A VECTOR OF POINTS, RANDOMIZED WITH PER DIMENSION RANDOM NUMBERS
            std::vector<boost::normal_distribution<double>> distributions;
            distributions.reserve(config.numDimensions);

            // and actually create them
            auto clusterCentroidGeneratorDistributionView = std::ranges::views::iota(static_cast<size_t>(0), config.numDimensions)
                | std::ranges::views::transform([&config](size_t dimension) {
                    PROFILE_FUNCTION();
                    return std::uniform_real_distribution<double>(config.clusterDimensionDistributions[dimension].low, config.clusterDimensionDistributions[dimension].high);
                });

            // and expand their pipeline
            //auto clusterCentroidGeneratorDistribution = std::vector<std::uniform_real_distribution<double>>(clusterCentroidGeneratorDistributionView.begin(), clusterCentroidGeneratorDistributionView.end());
            std::vector<std::uniform_real_distribution<double>> clusterCentroidGeneratorDistribution;
            clusterCentroidGeneratorDistribution.reserve(config.numDimensions);
            std::ranges::move(clusterCentroidGeneratorDistributionView, std::back_inserter(clusterCentroidGeneratorDistribution));

            // now, we can create our *actual* centroids with them
            // We are not using pipelining for this one since the generation is inherently stateful and thus problematic
            // if we attempt to assemble a data pipeline. Trust me, it makes your head explode
            m_KnownGoodCentroids = std::vector<Point>();
            m_KnownGoodCentroids.value().reserve(config.numTrueClusters);
            for (size_t sample = 0; sample < config.numTrueClusters; ++sample) {
                std::vector<double> coordinates;
                coordinates.reserve(config.numDimensions);
                for (auto& dist : clusterCentroidGeneratorDistribution) {
                    coordinates.push_back(dist(*rng));
                }
                m_KnownGoodCentroids.value().emplace_back(coordinates);
            }
        }

        // now that we have known good centroids FROM WHICH we can generate our clusters, we can actually generate the cluster
        // First we need to get a list of the NUMBER of samples per centroid
        // In other words, the "stride" of the data
        size_t samplesPerCentroid = config.numTotalSamples / config.numTrueClusters;
        size_t samplesLeftover = config.numTotalSamples % config.numTrueClusters;
        // Create a view for the number of samples per cluster, accounting for leftovers.
        auto samplesPerClusterView = std::ranges::views::iota(static_cast<size_t>(0), config.numTrueClusters)
            | std::ranges::views::transform([samplesPerCentroid, samplesLeftover](size_t clusterIdx) {
                PROFILE_FUNCTION();
                // Distribute leftovers to the first few clusters.
                return samplesPerCentroid + (clusterIdx < samplesLeftover ? 1 : 0);
            });

        // Now that we know how many per centroid, we can call DataSet::generateCluster for each centroid, with the number of samples, the RNG, the cluster center as the centroid.
        // Generate clusters using a ranges pipeline.

        auto knownGoodCentroidsWithCountsView = std::ranges::views::zip(m_KnownGoodCentroids.value(), samplesPerClusterView);

        auto clustersView = knownGoodCentroidsWithCountsView
            | std::ranges::views::transform([rng, &config](const auto& tuple) {
                PROFILE_FUNCTION();

                const auto& centroid = std::get<0>(tuple); // get the centroid
                const size_t numSamples = std::get<1>(tuple); // and the number of samples
                return generateCluster(
                    centroid,
                    numSamples,
                    config.clusterSpread,
                    rng
                );
            })
            | std::ranges::views::join; // Flatten the clusters into a single range of points.

        // Collect the generated points into m_Points
        {
            PROFILE_SCOPE("Collect generated points into m_Points");
            m_Points.reserve(config.numTotalSamples);
            std::ranges::move(clustersView, std::back_inserter(m_Points));
        }
    }

    Point DataSet::generateSinglePoint(const std::shared_ptr<std::vector<boost::normal_distribution<double>>>& distributions, std::shared_ptr<std::mt19937> rng) {
        PROFILE_FUNCTION();

        // back inserter doesn't play nice so we will make a vector then move it to the
        std::vector<double> dimensionsForPoint;
        dimensionsForPoint.reserve(distributions->size()); // Pre-reserve for dimensions.

        std::ranges::transform(*distributions,
                               std::back_inserter(dimensionsForPoint),
                               [&](auto& dist){ return dist(*rng); }); // Sample each dimension

        return Point(std::move(dimensionsForPoint)); // Assuming Point can be constructed from std::vector<double>.
        // If Point *is* std::vector<double>, just 'return dimensionsForPoint;'

    }


} // kmeans