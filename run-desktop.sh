#!/bin/bash

# --- LOCAL MACBOOK BENCHMARK SCRIPT ---
# This script is designed for local execution on a MacBook.
# It uses significantly reduced dataset sizes to prevent excessive memory usage and thrashing.

# --- IMPORTANT LOCAL SETUP NOTES ---
# 1. Ensure CMake, Make, a C++ compiler (like g++), and a local MPI implementation
#    (e.g., Open MPI installed via Homebrew) are installed and accessible in your PATH.
# 2. Ensure Boost libraries (especially Boost.MPI and Boost.Serialization) are installed
#    and your compiler/linker can find them.
# 3. This script does NOT handle module loading (Lmod is an HPC-specific tool).
# --- END LOCAL SETUP NOTES ---

# Define build directory and executable path
BUILD_DIR="build"
EXECUTABLE="./${BUILD_DIR}/kmeans_mpi"

# --- Build process ---
# Navigate to the directory where the script is run (assuming it's the repository root)
# And then proceed to build.
echo "--- Building kmeans_mpi executable ---"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}" || { echo "Error: Failed to enter build directory."; exit 1; }
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++ -G "Unix Makefiles" ..
make -j$(sysctl -n hw.ncpu) # Use all available CPU cores for compilation
cd .. # Go back to the root directory

# --- K-Means Parameters for MacBook-safe execution ---
# These parameters are much smaller than your OSC runs to prevent memory thrashing.
# Adjust --num-samples, --dimensions, --clusters further if your MacBook still struggles.
MAX_ITERATIONS=100000
NUM_SAMPLES=1000000        # Drastically reduced from 1M
DIMENSIONS=25             # Drastically reduced from 25
CLUSTERS=25              # Reduced from 50
SPREAD=3.5
SEED=1234
TRIALS=10 # Still run 10 trials for local averaging, but on smaller data

# --- Print header to CSV (using mpiexec -n 1 for consistency) ---
echo "--- Printing CSV Header ---"
mpiexec -n 1 "${EXECUTABLE}" --print-header --max-iterations "${MAX_ITERATIONS}" --num-samples "${NUM_SAMPLES}" --dimensions "${DIMENSIONS}" --clusters "${CLUSTERS}" --spread "${SPREAD}" --trials 10

# --- Run benchmarks for process counts 1-10 ---
echo "--- Running Benchmarks ---"
for num_procs in 1 2 3 4 5 6 7 8 9 10; do
    echo "Running with ${num_procs} processes..."
    # mpiexec -n <num_processes> <executable> [kmeans_args]
    mpiexec -n "${num_procs}" "${EXECUTABLE}" \
        --max-iterations "${MAX_ITERATIONS}" \
        --num-samples "${NUM_SAMPLES}" \
        --dimensions "${DIMENSIONS}" \
        --clusters "${CLUSTERS}" \
        --spread "${SPREAD}" \
        --seed "${SEED}" \
        --trials "${TRIALS}";
done

echo "--- Benchmarking complete. Results saved ---"