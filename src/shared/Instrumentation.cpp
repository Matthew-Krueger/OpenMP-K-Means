//
// Created by Matthew Krueger on 10/14/25.
//

#include "Instrumentation.hpp"

#include <algorithm>
#include <fstream>

//#define DEBUG_INSTRUMENTATION


#ifdef DEBUG_INSTRUMENTATION
#define INSTRUMENTATION_DEBUG_INSTRUMENTATION true
#else
#define INSTRUMENTATION_DEBUG_INSTRUMENTATION false
#endif

namespace instrumentation {
    std::shared_ptr<Instrumentor> Instrumentor::s_GlobalInstrumentor = nullptr;

    std::weak_ptr<Instrumentor> Instrumentor::getGlobalInstrumentor() {
        return Instrumentor::s_GlobalInstrumentor;
    }

    Instrumentor::Instrumentor(std::unique_ptr<Writer> &&writer) {
        m_Writer = std::move(writer);
    }

    void Instrumentor::recordEntry(Entry &&entry) {
        m_LocalLog.push_back(std::move(entry));

        if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
            std::cout << "Wrote an instrumentation entry. Local Log Size: " << m_LocalLog.size() << " entries" <<
                    std::endl;
        }

        if (m_LocalLog.size() > m_Writer->getTargetBufferSize()) {
            flush();
        }
    }

    Instrumentor::~Instrumentor() {
        flush();
        m_Writer->flush();
        // flush the writer too since it will go out of scope when this does. The writer will finalize itself on destruct however since it is not a global singleton
    }


    void Instrumentor::flush() {
        if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
            std::cout << "Flushing Instrumentation Log" << std::endl;
        }
        m_Writer->write(m_LocalLog);
        m_LocalLog.clear();
    }

    void Instrumentor::initializeGlobalInstrumentor(std::unique_ptr<Writer> &&writer) {
        if (s_GlobalInstrumentor == nullptr) {
            s_GlobalInstrumentor = std::make_shared<Instrumentor>(std::move(writer));
        } else {
            std::cout << "Global Instrumentor already exists. Ignoring request." << std::endl;
        }
    }

    void Instrumentor::finalizeGlobalInstrumentor() {
        s_GlobalInstrumentor.reset(); // invalidate *all* instances of this instrumentor
    }
}
#ifdef BUILD_WITH_MPI
#include <mpi.h>
#endif

namespace instrumentation {

    std::variant<std::chrono::time_point<std::chrono::high_resolution_clock>, double> Session::getTimePoint() {
        if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
            std::cout << "Getting time point" << std::endl;
        }

        #ifdef BUILD_WITH_MPI
        return MPI_Wtime();
        #else
        return std::chrono::high_resolution_clock::now();
        #endif
    }

#ifdef BUILD_WITH_MPI
    MPIWriter::MPIWriter(const Config& config) : Writer(config.targetBufferSize) {
        m_MyRank = std::numeric_limits<int>::max();
        m_Config = config;

        MPI_Comm_rank(MPI_COMM_WORLD, &m_MyRank);
        MPI_Comm_size(MPI_COMM_WORLD, &m_WorldSize);

        m_WriteBuffer.reserve(config.targetBufferSize * 2); // Pre-allocate
        m_Displacements.reserve(config.targetBufferSize);
        m_EntrySizes.reserve(config.targetBufferSize);

        m_IsFirstFlush = true;
    }


    MPIWriter::~MPIWriter() {
        flush();
        if (m_MyRank == m_Config.mainRank) {
                if (std::ofstream file(m_Config.logFileName, std::ios::app); file.is_open()) {
                    write_tail(file);
                }

        }
    }

    void MPIWriter::write(const std::vector<Entry>& entries) {
        if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
            std::cout << "Writing an MPIWriter entry." << std::endl;
        }

        // Append entries to m_WriteBuffer and track their starting positions and sizes
        for (const auto& entry : entries) {
            const std::string& entryText = entry.to_string();
            if (!entryText.empty()) {
                m_Displacements.push_back(static_cast<uint32_t>(m_WriteBuffer.size()));
                m_EntrySizes.push_back(static_cast<uint32_t>(entryText.size()));

                size_t oldSize = m_WriteBuffer.size();
                m_WriteBuffer.resize(oldSize + entryText.size());
                std::copy(entryText.begin(), entryText.end(), m_WriteBuffer.begin() + oldSize);
            }
        }

        if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
            std::cout << "Wrote an MPIWriter entry. Local Log Size: " << m_WriteBuffer.size() << " characters" << std::endl;
            std::cout << "Displacements: " << m_Displacements.size() << std::endl;
            std::cout << "Contents: ";
            std::cout.write(m_WriteBuffer.data(), m_WriteBuffer.size());
            std::cout << std::endl;
        }
    }

    // Flush function
    void MPIWriter::flush() {
        if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
            std::cout << "Flushing MPIWriter. Local buffer size: " << m_WriteBuffer.size() << std::endl;
        }

        int local_size = static_cast<int>(m_WriteBuffer.size());
        int local_entry_count = static_cast<int>(m_EntrySizes.size());

        if (m_MyRank == m_Config.mainRank) {
            if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
                std::cout << "Main rank flushing" << std::endl;
            }

            // Main rank: receive from all processes and write to file
            std::vector<int> recv_counts(m_WorldSize, 0);
            std::vector<int> entry_counts(m_WorldSize, 0);

            // Gather total character counts and entry counts from all ranks
            MPI_Gather(&local_size, 1, MPI_INT,
                       recv_counts.data(), 1, MPI_INT,
                       m_Config.mainRank, MPI_COMM_WORLD);
            MPI_Gather(&local_entry_count, 1, MPI_INT,
                       entry_counts.data(), 1, MPI_INT,
                       m_Config.mainRank, MPI_COMM_WORLD);

            // Calculate total size and displacements for data
            std::vector<int> data_displacements(m_WorldSize, 0);
            int total_size = 0;
            for (int i = 0; i < m_WorldSize; ++i) {
                data_displacements[i] = total_size;
                total_size += recv_counts[i];
            }

            // Calculate total entries and displacements for entry sizes
            int total_entries = 0;
            std::vector<int> entry_displacements(m_WorldSize, 0);
            for (int i = 0; i < m_WorldSize; ++i) {
                entry_displacements[i] = total_entries;
                total_entries += entry_counts[i];
            }

            // Allocate receive buffers
            std::vector<char> recv_buffer(total_size);
            std::vector<int> all_entry_sizes(total_entries);

            // Gather entry sizes
            MPI_Gatherv(m_EntrySizes.data(), local_entry_count, MPI_INT,
                        all_entry_sizes.data(), entry_counts.data(), entry_displacements.data(), MPI_INT,
                        m_Config.mainRank, MPI_COMM_WORLD);

            // Gather actual data
            MPI_Gatherv(m_WriteBuffer.data(), local_size, MPI_CHAR,
                        recv_buffer.data(), recv_counts.data(), data_displacements.data(), MPI_CHAR,
                        m_Config.mainRank, MPI_COMM_WORLD);

            // Write to the appropriate destination
            if (std::ofstream file(m_Config.logFileName, std::ios::app); file.is_open()) {
                    if (m_IsFirstFlush) {
                        write_preamble(file);
                        m_IsFirstFlush = false;
                    }
                    writeEntriesToFile(recv_buffer, all_entry_sizes, file);
                    file.flush();
                }

            if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
                std::cout << "Main rank finished flushing" << std::endl;
            }

        } else {
            if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
                std::cout << "Non-main rank flushing" << std::endl;
            }

            // Non-main ranks: send data to main rank
            MPI_Gather(&local_size, 1, MPI_INT,
                       nullptr, 0, MPI_INT,
                       m_Config.mainRank, MPI_COMM_WORLD);
            MPI_Gather(&local_entry_count, 1, MPI_INT,
                       nullptr, 0, MPI_INT,
                       m_Config.mainRank, MPI_COMM_WORLD);

            if (local_size > 0) {
                MPI_Gatherv(m_EntrySizes.data(), local_entry_count, MPI_INT,
                            nullptr, nullptr, nullptr, MPI_INT,
                            m_Config.mainRank, MPI_COMM_WORLD);
                MPI_Gatherv(m_WriteBuffer.data(), local_size, MPI_CHAR,
                            nullptr, nullptr, nullptr, MPI_CHAR,
                            m_Config.mainRank, MPI_COMM_WORLD);
            }

            if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
                std::cout << "Non-main rank finished flushing" << std::endl;
            }
        }

        // Ensure all ranks complete communication before clearing buffers
        MPI_Barrier(MPI_COMM_WORLD);

        // Clear buffers after flush
        m_WriteBuffer.clear();
        m_Displacements.clear();
        m_EntrySizes.clear();
    }

    void MPIWriter::writeEntriesToFile(const std::vector<char>& all_data, const std::vector<int>& entry_sizes, std::ostream& file) {
        if (all_data.empty() || entry_sizes.empty()) {
            return;
        }

        bool first_entry = true;
        size_t offset = 0;

        for (size_t i = 0; i < entry_sizes.size(); ++i) {
            if (entry_sizes[i] == 0) {
                continue; // Skip empty entries
            }
            if (offset + entry_sizes[i] > all_data.size()) {
                break; // Prevent buffer overrun
            }

            if (!first_entry) {
                file << COMMA_NEWLINE;
            }
            file.write(all_data.data() + offset, entry_sizes[i]);
            first_entry = false;
            offset += entry_sizes[i];
        }
    }

    uint32_t MPIWriter::getProcessID() {
        return m_MyRank;
    }


#endif
}
