//
// Created by Matthew Krueger on 10/13/25.
//

#ifndef KMEANS_MPI_OMPSOLVER_HPP
#define KMEANS_MPI_OMPSOLVER_HPP

#include "../shared/DataSet.hpp"

namespace kmeans {
    class OpenMPSolver {
    public:
        struct Config {
            size_t maxIterations;
            double convergenceThreshold;
            DataSet dataSet;
            size_t startingCentroidSeed;
            size_t startingCentroidCount;
            size_t numThreads;
        };

        OpenMPSolver() = default;
        OpenMPSolver(Config &config);
        OpenMPSolver(const OpenMPSolver&) = delete;
        OpenMPSolver(Config &&config) = delete;
        OpenMPSolver& operator=(const OpenMPSolver&) = delete;
        OpenMPSolver& operator=(OpenMPSolver&&) = delete;
        ~OpenMPSolver() = default;

        void run();

        inline DataSet& getDataSet() { return m_DataSet; }
        inline const DataSet& getDataSet() const { return m_DataSet; }
        inline const std::optional<std::vector<Point>>& getCalculatedCentroidsAtCompletion() const { return m_CalculatedCentroidsAtCompletion; }

        inline const std::optional<size_t>& getFinalIterationCount() const { return m_FinalIterationCount; }

    private:
        DataSet m_DataSet;
        std::vector<Point> m_CurrentCentroids;
        std::vector<Point> m_PreviousCentroids;
        size_t m_MaxIterations;
        size_t m_NumThreads;
        double m_ConvergenceThreshold;
        std::optional<std::vector<Point>> m_CalculatedCentroidsAtCompletion = std::nullopt;
        std::optional<size_t> m_FinalIterationCount = std::nullopt;


    };
} // kmeans

#endif //KMEANS_MPI_OMPSOLVER_HPP