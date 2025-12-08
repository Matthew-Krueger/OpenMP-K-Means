//
// Created by Matthew Krueger on 10/14/25.
//

#ifndef KMEANS_MPI_INSTRUMENTATION_HPP
#define KMEANS_MPI_INSTRUMENTATION_HPP

#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <thread>
#include <variant>
#include <cstring>
#include <sstream>
#include <functional>

//#define BUILD_WITH_PROFILING


namespace instrumentation {

    inline const char* cStringEscape(const char* toEscape){

        static char buffer[4096];

        strcpy(buffer, toEscape);
        size_t originalIndex = 0;
        size_t originalSize = strlen(toEscape);
        size_t index = 0;
        for(originalIndex = 0; originalIndex<originalSize; ++originalIndex){
            char currentChar = toEscape[originalIndex];

            if(toEscape[originalIndex] != buffer[index]){
                exit(-1);
            }

            if(currentChar == '\"'){
                buffer[index] = '\\';
                ++index;
                buffer[index] = '\0';
                // now copy the string from this point onwards
                strcat(&buffer[index],&toEscape[originalIndex]);
            }

            // remember to increment the original index
            ++index;
        }

        return buffer;

    }



    class Entry {
    public:
        struct ProfileResult{
            const char* name;
            long long start, end;
            uint32_t threadID;
            uint32_t processID;
        };

        /**
         * @brief A wrapper for a string representing a log entry.
         *
         * This is specifically a wrapper IN ORDER THAT it may be extendable in the future.
         * @param m_Value A serialized version of the log entry. In the case of Chrome Tracer logs, this is the log without a comma
         */
        explicit Entry(std::string m_Value) : m_Value(std::move(m_Value)){};
        Entry(Entry&& other) noexcept = default;
        Entry(const Entry& other) = default;
        Entry(const ProfileResult& result) { // NOLINT(*-explicit-constructor)

            const char* name = cStringEscape(result.name);

            std::ostringstream oss;
            oss << "{";
            oss << R"("cat":"function",)";
            oss << "\"dur\":" << (result.end - result.start) << ',';
            oss << R"("name":")" << name << "\",";
            oss << R"("ph":"X",)";
            oss << "\"pid\":" << result.processID << ",";
            oss << "\"tid\":" << result.threadID << ",";
            oss << "\"ts\":" << result.start;
            oss << "}";

            m_Value = oss.str();
        }

        ~Entry() = default;

        Entry& operator=(const Entry&) = default;
        Entry& operator=(Entry&&) = default;
        Entry& operator=(std::string&&) = delete;

        inline const std::string& operator*() const{ return m_Value; }

        [[nodiscard]] inline const std::string& to_string() const {
            return m_Value;
        };

        friend std::ostream& operator<<(std::ostream& os, const Entry& entry) { os << entry.m_Value; return os; }

    private:
        std::string m_Value;
    };

    class Writer{
    public:

        explicit Writer(size_t targetBufferSize) : m_TargetBufferSize(targetBufferSize) {};
        virtual ~Writer() {}

        virtual void write(const std::vector<Entry>& entries) = 0;
        virtual uint32_t getThreadID() { return std::hash<std::thread::id>{}(std::this_thread::get_id()); }
        virtual uint32_t getProcessID() { return 0; }

        /**
         *  @brief Flushes the buffer to the disk.
         *
         * In distributed memory environments, dysfunction will also synchronize everything to the main thread.
         * This means it is imperative to call this manually or at destruction, as it will block.
         */
        virtual void flush() = 0;

        [[nodiscard]] inline size_t getTargetBufferSize() const { return m_TargetBufferSize; }

    private:
        size_t m_TargetBufferSize;

    };


#ifdef BUILD_WITH_MPI

#endif

    class Instrumentor {
    public:
        explicit Instrumentor(std::unique_ptr<Writer>&& writer);
        Instrumentor(const Instrumentor&) = delete;
        Instrumentor(Instrumentor &&) = delete;
        ~Instrumentor();

        Instrumentor& operator=(const Instrumentor&) = delete;
        Instrumentor& operator=(Instrumentor&&) = delete;
        Instrumentor& operator=(std::unique_ptr<Writer>&& writer) = delete;

        void recordEntry(Entry&& entry);
        void flush();
        std::unique_ptr<Writer>& getWriter() { return m_Writer; }

        static std::weak_ptr<Instrumentor> getGlobalInstrumentor();
        static void initializeGlobalInstrumentor(std::unique_ptr<Writer>&& writer);
        static void finalizeGlobalInstrumentor();

    private:


        static std::shared_ptr<Instrumentor> s_GlobalInstrumentor;

        std::unique_ptr<Writer> m_Writer;
        std::vector<Entry> m_LocalLog;

    };

    class Session {
    public:
        inline explicit Session(const char* name) : m_Name(name), m_StartTime(getTimePoint()), m_Stopped(false) {};
        ~Session() {
            stop();
        };

        inline void stop(){
            if (!m_Stopped) {
                auto endTimepoint = getTimePoint();

                long long end = convertTimepointToMicroseconds(endTimepoint);

                if (auto instrumentor = Instrumentor::getGlobalInstrumentor().lock()) {
                    uint32_t threadID = instrumentor->getWriter()->getThreadID();
                    uint32_t processID = instrumentor->getWriter()->getProcessID();
                    instrumentor->recordEntry(Entry::ProfileResult{m_Name, convertTimepointToMicroseconds(m_StartTime), end, threadID, processID});
                }

                m_Stopped = true;
            }
        }


    private:
        inline static long long convertTimepointToMicroseconds(std::variant<std::chrono::time_point<std::chrono::high_resolution_clock>, double> startTimepoint) {
            if (std::holds_alternative<double>(startTimepoint)) {
                // we are using MPI for the session
                // convert seconds to microseconds
                return static_cast<long long>(std::get<double>(startTimepoint) * 1e6);
            }else if (std::holds_alternative<std::chrono::time_point<std::chrono::high_resolution_clock>>(startTimepoint)) {
                return std::chrono::time_point_cast<std::chrono::microseconds>(std::get<std::chrono::time_point<std::chrono::high_resolution_clock>>(startTimepoint)).time_since_epoch().count();
            }else {
                throw std::runtime_error("Unknown timepoint type");
            }
        }

        static std::variant<std::chrono::time_point<std::chrono::high_resolution_clock>, double> getTimePoint();

        const char* m_Name;
        std::variant<std::chrono::time_point<std::chrono::high_resolution_clock>, double> m_StartTime;
        bool m_Stopped;

    };

#ifdef BUILD_WITH_MPI
    class MPIWriter final : public Writer {
    public:
        struct Config{
            std::string logFileName;
            int mainRank;
            int logTag;
            size_t targetBufferSize;
        };

        explicit MPIWriter(const Config& config);
        ~MPIWriter() override;

        void write(const std::vector<Entry>& entries) override;
        void flush() override;

        uint32_t getProcessID() override;

    private:
        Config m_Config;
        std::vector<char> m_WriteBuffer;      // Stores concatenated entry strings as chars
        std::vector<uint32_t> m_Displacements; // Tracks start position of each entry
        std::vector<uint32_t> m_EntrySizes;   // Tracks size of each entry
        int m_MyRank;
        int m_WorldSize;
        bool m_IsFirstFlush = true;
        static constexpr const char* PREAMBLE = "[\n";
        static constexpr const char* TAIL = "\n]";
        static constexpr const char* COMMA_NEWLINE = ",\n";

        static void write_preamble(std::ostream& file) {
            file << PREAMBLE;
        }

        static void write_tail(std::ostream& file) {
            file << TAIL;
        }

        static void writeEntriesToFile(const std::vector<char>& all_data, const std::vector<int>& global_displacements, std::ostream& file);
    };
#endif


}
#ifdef BUILD_WITH_PROFILING
#ifdef __GNUC__
#define __PROFILE_FUNCTION_NAME                    __PRETTY_FUNCTION__
#else
#define _PROFILE_FUNCTION_NAME                    __FUNCSIG__
#endif
#define __PROFILE_TIMER_NAME                       timer
#define PROFILE_BEGIN_SESSION(writer)             ::instrumentation::Instrumentor::initializeGlobalInstrumentor(writer)
#define PROFILE_END_SESSION()                     ::instrumentation::Instrumentor::finalizeGlobalInstrumentor()
#define PROFILE_SCOPE(name)                       ::instrumentation::Session session##__LINE__(name)
#define PROFILE_FUNCTION()                        PROFILE_SCOPE(__PROFILE_FUNCTION_NAME)
#else
#define PROFILE_BEGIN_SESSION(writer)             (void(0))
#define PROFILE_END_SESSION()                     (void(0))
#define PROFILE_FUNCTION()                        (void(0))
#define PROFILE_SCOPE(name)                       (void(0))
#endif

#endif //KMEANS_MPI_INSTRUMENTATION_HPP