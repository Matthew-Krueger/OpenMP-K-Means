cmake_minimum_required(VERSION 3.25)
project(kmeans_openmp)

set(CMAKE_CXX_STANDARD 23)
option(BUILD_WITH_MPI "Build with MPI support" OFF)
option(USE_OpenMP "Enable OpenMP support" ON)

if(BUILD_WITH_MPI)
    add_compile_definitions(BUILD_WITH_MPI)
    set(MPI_SOURCES "src/mpi/MPISolver.cpp"
                    "src/mpi/MPISolver.hpp"
                    "src/mpi/MPITester.cpp"
                    "src/mpi/MPITester.hpp")
    find_package(MPI REQUIRED)
endif(BUILD_WITH_MPI)
if(USE_OpenMP)
    set(MANUAL_OMP_CONFIG FALSE)

    if(APPLE)
        execute_process(COMMAND brew --prefix libomp
                OUTPUT_VARIABLE LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
        if(LIBOMP_PREFIX)
            set(OMP_INCLUDE_DIR "${LIBOMP_PREFIX}/include")
            set(OMP_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${OMP_INCLUDE_DIR}")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${OMP_INCLUDE_DIR}")
            set(OpenMP_C_INCLUDE_DIR "${OMP_INCLUDE_DIR}")
            set(OpenMP_CXX_INCLUDE_DIR "${OMP_INCLUDE_DIR}")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${OMP_LIBRARY}")
            set(OpenMP_FOUND TRUE)
            set(MANUAL_OMP_CONFIG TRUE)
            message(STATUS "Manually configured OpenMP for macOS at ${LIBOMP_PREFIX}")
        endif()
    endif()

    if(NOT MANUAL_OMP_CONFIG)
        find_package(OpenMP QUIET)
        if(NOT OpenMP_FOUND)
            message(WARNING "OpenMP not found. Compiling without OpenMP support.")
            set(USE_OpenMP OFF)
        endif()
    endif()

    if(OpenMP_FOUND)
        message(STATUS "OpenMP found: C flags = ${OpenMP_C_FLAGS}, CXX flags = ${OpenMP_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# For GCC or Clang: -O3 enables auto-vectorization (implies -ftree-vectorize),
# -march=native uses all available CPU extensions (e.g., AVX, SSE) on the build machine.
#add_compile_options(-O3 -march=native -ftree-vectorize -flto -funroll-loops -std=c++23)
add_compile_options(-Wunknown-pragmas)



#cmake_policy(SET CMP0144 NEW)
#cmake_policy(SET CMP0167 OLD)
set(BOOST_ROOT /boost/boost_1_82_0)

if(BUILD_WITH_MPI)
    find_package(Boost 1.82.0 COMPONENTS mpi serialization program_options random REQUIRED)
else(BUILD_WITH_MPI)
    find_package(Boost 1.82.0 COMPONENTS serialization program_options random REQUIRED)
endif(BUILD_WITH_MPI)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})


add_executable(kmeans_openmp
        src/shared/Point.cpp
        src/shared/Point.hpp
        src/shared/DataSet.cpp
        src/shared/DataSet.hpp
        src/serial/SerialSolver.cpp
        src/serial/SerialSolver.hpp
        src/shared/Utils.cpp
        src/shared/Utils.hpp
        src/shared/Instrumentation.cpp
        src/shared/Instrumentation.hpp
        src/shared/Timer.cpp
        src/shared/Timer.hpp
        src/shared/DualOutputStream.hpp
        src/normal-main.cpp
        src/omp/OpenMPSolver.cpp
        src/omp/OpenMPSolver.hpp
        src/shared/CentroidValidator.hpp
        ${MPI_SOURCES}

)

if(BUILD_WITH_MPI)
    target_link_libraries(kmeans_openmp PRIVATE MPI::MPI_CXX ${Boost_LIBRARIES})
else()
    target_link_libraries(kmeans_openmp PRIVATE ${Boost_LIBRARIES})
endif()

if(USE_OpenMP AND OpenMP_FOUND)
    if(MANUAL_OMP_CONFIG)
        # Manual linking for macOS libomp
        target_link_libraries(kmeans_openmp PRIVATE ${OMP_LIBRARY})
        target_include_directories(kmeans_openmp PRIVATE ${OMP_INCLUDE_DIR})
    else()
        # Standard linking via imported target
        target_link_libraries(kmeans_openmp PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()