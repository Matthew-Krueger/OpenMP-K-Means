//
// Created by Matthew Krueger on 10/17/25.
//

#include "Timer.hpp"

#include <chrono>
#include <iostream>

#undef BUILD_WITH_MPI_FLAG
#ifdef BUILD_WITH_MPI
#define BUILD_WITH_MPI_FLAG true
#else
#define BUILD_WITH_MPI_FLAG false
double MPI_Wtime(){ return 0.0; };
#endif

#ifdef BUILD_WITH_MPI
#include <mpi.h>
#endif

namespace timer {
    Timer::Timer(std::weak_ptr<uint64_t> timeReference) : m_TimeReference(std::move(timeReference)){
        if constexpr (BUILD_WITH_MPI_FLAG) {
            m_StartTimePoint = MPI_Wtime();
        }else {
            m_StartTimePoint = static_cast<uint64_t>(std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count());
        }
    }

    Timer::~Timer() {

        // get our end time
        uint64_t endTime = 0;
        if constexpr (BUILD_WITH_MPI_FLAG) {
            endTime = static_cast<uint64_t>(MPI_Wtime()*1e6);
        }else {
            endTime = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
        }

        uint64_t startTime = 0;
        if (std::holds_alternative<uint64_t>(m_StartTimePoint)) {
            startTime = std::get<uint64_t>(m_StartTimePoint);
        } else if(std::holds_alternative<double>(m_StartTimePoint)) {
            startTime = std::get<double>(m_StartTimePoint)*1e6; // convert seconds to microseconds
        }else {
            std::cerr << "Unknown timepoint type" << std::endl;
            //throw std::runtime_error("Unknown timepoint type");
        }

        if (auto timeReference = m_TimeReference.lock()) {
            *timeReference = endTime - startTime; // get the difference and write it to our reference
        } else {
            std::cerr << "Unable to get time reference" << std::endl;
            //throw std::runtime_error("Unable to get time reference");
        }

    }

}
