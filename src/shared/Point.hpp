//
// Created by mpiuser on 10/10/25.
// This file defines the Point class, which represents a data point in a multi-dimensional space.

#ifndef KMEANS_MPI_POINT_HPP
#define KMEANS_MPI_POINT_HPP
#include <utility>
#include <vector>
#include <cstddef>
#include <string>
#include <iostream>
#include <boost/serialization/access.hpp>

namespace kmeans{
    class Point{
    public:

        /**
         * @brief A structure to hold flattened point data for efficient serialization/deserialization.
         *
         * This structure is used to represent a collection of points as a single contiguous
         * vector of doubles, along with metadata about the original dimensions and number of points.
         */
        struct FlattenedPoints
        {
            /// The number of dimensions for each point.
            size_t numDimensionsPerPoint;
            /// The total number of points.
            size_t numPoints;
            /// A flattened vector containing all the double values of all points.
            /// The data for each point is stored contiguously.
            std::vector<double> points;
        };

        Point() = default;

        /**
         *
         * @param data Data to store
         * @param count Internal variable. DO not touch except to store hidden data
         */
        explicit Point(std::vector<double> data, size_t count = 1) noexcept: m_Data(std::move(data)), m_Count(count) {}

        /**
         * @brief Copy constructor.
         * @param other The Point object to copy from.
         */
        Point(const Point& other) = default;

        /**
         * @brief Move constructor.
         * @param other The Point object to move from.
         */
        Point(Point&& other) noexcept : m_Data{std::move(other.m_Data)}, m_Count(other.m_Count){}

        /**
         * @brief Copy-and-swap assignment operator.
         * @param other The Point object to assign from.
         * @return A reference to the assigned Point object.
         */
        Point& operator=(Point other){
            std::swap(m_Data, other.m_Data);
            std::swap(m_Count, other.m_Count);
            return *this;
        }

        /**
         * @brief Default destructor.
         */
        ~Point() = default;

        /**
         * @brief Gets the underlying data vector of the point.
         * @return A const reference to the internal std::vector<double> representing the point's coordinates.
         */
        [[nodiscard]] const std::vector<double>& getData() const { return m_Data; };

        /**
         * @brief Gets the underlying data vector of the point.
         * @return A reference to the internal std::vector<double> representing the point's coordinates.
         */
        std::vector<double>& getData() { return m_Data; };

        /**
         * @brief Sets the underlying data vector of the point.
         * @param data A std::vector<double> to set as the point's coordinates.
         */
        void setData(std::vector<double> data) { m_Data = std::move(data); };

        /**
         * @brief Gets the number of dimensions in the point.
         * @return The number of dimensions in the point.
         */
        [[nodiscard]] inline size_t numDimensions() const { return m_Data.size(); }

        /**
         * @brief Gets the value of a specific dimension in the point.
         * @param index The index of the dimension to get.
         * @return The value of the dimension at the specified index.
         */
        [[nodiscard]] inline double operator[](const size_t index) const { return m_Data[index]; }

        /**
         * @brief Gets the number of elements in the data vector representing the point.
         *
         * This method returns the total number of values stored in the internal data vector,
         * which corresponds to the number of dimensions of the point.
         *
         * @return The size of the data vector as a size_t.
         */
        inline size_t size() const { return m_Data.size(); }

        /**
         * @brief Checks if the point's data vector is empty.
         *
         * This method determines whether the internal data vector of the point contains
         * any elements, effectively checking if the point has any dimensions defined.
         *
         * @return True if the data vector is empty, otherwise false.
         */
        inline bool empty() const { return m_Data.empty(); }

        /**
         * @brief Calculates the Euclidean distance between this point and another point.
         * @param other The other point to calculate the distance to.
         * @return Euclidean distance as a double
         */
        double calculateEuclideanDistance(const Point& other) const;

        static FlattenedPoints flattenPoints(const std::vector<Point>& points);
        static std::vector<Point> unflattenPoints(const FlattenedPoints &flattenedPoints);

        Point& operator+=(const Point& other);
        Point& operator/=(double scalar);

        Point operator+(const Point& other) const {
            Point result = *this;
            result += other;
            return result;
        }

        Point operator/(const double scalar) const {
            Point result = *this;
            result /= scalar;
            return result;
        }

        // Iterator support to allow for range-based for loops over the data, among other syntax magic
        using iterator = std::vector<double>::iterator;
        using const_iterator = std::vector<double>::const_iterator;

        inline iterator begin() { return m_Data.begin(); }
        [[nodiscard]] inline const_iterator begin() const { return m_Data.begin(); }
        inline iterator end() { return m_Data.end(); }
        [[nodiscard]] inline const_iterator end() const { return m_Data.end(); }


        [[nodiscard]] std::vector<Point>::iterator findClosestPointInVector(std::vector<Point>& other) const;

        inline size_t getCount() const { return m_Count; }
        inline void setCount(size_t count) { m_Count = count; }

        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version) {
            ar & m_Data;
            ar & m_Count;
        };

    private:
        std::vector<double> m_Data;
        size_t m_Count = 1;
    };

    inline std::ostream& operator<<(std::ostream& os, const Point& point) {

        os << "Point: (";

        bool first = true;
        for (auto demention:point) {
            if (first) {
                first = false;
            }else {
                os << ',';
            }
            os << demention;
        }
        return os << ')';

    }


    class ClusterLocalAggregateSum {
    public:
        ClusterLocalAggregateSum() = default;
        ClusterLocalAggregateSum(const ClusterLocalAggregateSum& other) = default;
        ClusterLocalAggregateSum(ClusterLocalAggregateSum&& other) = default;
        ClusterLocalAggregateSum& operator=(const ClusterLocalAggregateSum& other) = default;
        ClusterLocalAggregateSum& operator=(ClusterLocalAggregateSum&& other) = default;
        ~ClusterLocalAggregateSum() = default;

        ClusterLocalAggregateSum(Point localSumData, size_t localCount) : localSumData(std::move(localSumData)), localCount(localCount){}

        /**
         * The local sum of the point
         */
        Point localSumData{};
        /**
         * The local count (number of points that went into making the sum
         */
        size_t localCount{};

        template<class Archive>
        void serialize(Archive &ar, const unsigned int version) {
            ar &localSumData;
            ar &localCount;
        }

        static ClusterLocalAggregateSum calculateCentroidLocalSum(const std::vector<Point> &points);

    };

} // kmeans

#endif //KMEANS_MPI_POINT_HPP