//
// Created by Matthew Krueger on 10/17/25.
//

#ifndef KMEANS_MPI_UNIONOUTPUT_HPP
#define KMEANS_MPI_UNIONOUTPUT_HPP

#include <fstream>
#include <ostream>

class DualStream {
public:
    inline DualStream(std::ostream &output, std::string filename) : m_Output(output), m_File(filename, std::ios::out | std::ios::app) {
        if (!m_File.is_open()) {
            std::cerr << "Could not open file: " << filename << std::endl;
        }
    }

    inline ~DualStream() {
        m_File.close();
    }

    template<typename T>
    DualStream& operator<<(const T& val) {
        m_Output << val;
        if (m_File.is_open())
            m_File << val;
        return *this;
    }

    // Overload for manipulators like std::endl
    DualStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        manip(m_Output);
        if (m_File.is_open())
            manip(m_File);
        return *this;
    }


private:
    std::ostream &m_Output;
    std::ofstream m_File;

};

#endif //KMEANS_MPI_UNIONOUTPUT_HPP