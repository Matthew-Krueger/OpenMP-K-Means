//
// Created by Matthew Krueger on 10/15/25.
//

#ifndef KMEANS_MPI_MPISOLVER_HPP
#define KMEANS_MPI_MPISOLVER_HPP

#include <cstddef>
#include <boost/mpi/communicator.hpp>

#include "../shared/DataSet.hpp"

namespace kmeans {

    class MPISolver {
    public:
        struct Config {
            size_t maxIterations;
            double convergenceThreshold;
            DataSet dataSet;
            size_t startingCentroidSeed;
            size_t startingCentroidCount;
            int mainRank;
            int workingTag;

            Config() = delete;
            Config(size_t maxIterations, double convergenceThreshold, DataSet dataSet, size_t startingCentroidSeed, size_t startingCentroidCount, int mainRank, int workingTag) :
                    maxIterations(maxIterations),
                    convergenceThreshold(convergenceThreshold),
                    dataSet(std::move(dataSet)),
                    startingCentroidSeed(startingCentroidSeed),
                    startingCentroidCount(startingCentroidCount),
                    mainRank(mainRank),
                    workingTag(workingTag) {}
        };

        MPISolver() = delete;
        MPISolver(const MPISolver&) = delete;
        explicit MPISolver(Config &&config, boost::mpi::communicator& communicator);
        explicit MPISolver(Config &config) = delete;
        MPISolver& operator=(const MPISolver&) = delete;
        MPISolver& operator=(MPISolver&&) = delete;
        MPISolver(MPISolver&&) = default;
        ~MPISolver() = default;

        void run();

        void initialDistributeDataSet(DataSet && dataSet);
        void initialDistributeCentroids();

        void globalReduceCentroids();
        void globalGatherCentroids(const std::vector<Point> &localCentroids);
        static void applyScalarToCentroids(std::vector<Point> &centroids);


        inline std::optional<size_t> getFinalIterationCount() const { return m_FinalIterationCount; }
        inline const std::optional<std::vector<Point>>& getCalculatedCentroidsAtCompletion() const { return m_CalculatedCentroidsAtCompletion; }

    private:
        DataSet m_LocalDataSet;
        std::vector<Point> m_CurrentCentroids;
        std::vector<Point> m_PreviousCentroids;
        size_t m_MaxIterations;
        double m_ConvergenceThreshold;
        std::optional<std::vector<Point>> m_CalculatedCentroidsAtCompletion = std::nullopt;
        boost::mpi::communicator& m_Communicator;
        int m_MainRank;
        int m_WorkingTag;
        std::optional<size_t> m_FinalIterationCount = std::nullopt;


    };

}



#endif //KMEANS_MPI_MPISOLVER_HPP