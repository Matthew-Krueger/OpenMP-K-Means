//
// Created by Matthew Krueger on 10/13/25.
//

#ifndef KMEANS_MPI_SERIALSOLVER_HPP
#define KMEANS_MPI_SERIALSOLVER_HPP

#include "../shared/DataSet.hpp"

namespace kmeans {
    class SerialSolver {
    public:
        struct Config {
            size_t maxIterations;
            double convergenceThreshold;
            DataSet dataSet;
            size_t startingCentroidSeed;
            size_t startingCentroidCount;
        };

        SerialSolver() = default;
        SerialSolver(Config &config);
        SerialSolver(const SerialSolver&) = delete;
        SerialSolver(Config &&config) = delete;
        SerialSolver& operator=(const SerialSolver&) = delete;
        SerialSolver& operator=(SerialSolver&&) = delete;
        ~SerialSolver() = default;

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
        double m_ConvergenceThreshold;
        std::optional<std::vector<Point>> m_CalculatedCentroidsAtCompletion = std::nullopt;
        std::optional<size_t> m_FinalIterationCount = std::nullopt;


    };
} // kmeans

#endif //KMEANS_MPI_SERIALSOLVER_HPP