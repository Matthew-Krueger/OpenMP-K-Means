#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=3:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --account=PCS0294
#SBATCH --job-name=KMeansMPI

# Load modules
module load openmpi/5.0.2
module load boost/1.83.0
module load cmake/3.25.2
module load gcc/13.2.0

# Build process
cd "$SLURM_SUBMIT_DIR" || exit
mkdir -p build
cd build || exit
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++ -G "Unix Makefiles" ..
make

EXECUTABLE="$SLURM_SUBMIT_DIR/build/kmeans_mpi"

# Print header to CSV
srun --ntasks=1 "${EXECUTABLE}" --print-header

# Run benchmarks for thread counts 1-10
for num_omp_threads in 1 2 3 4 5 6 7 8 9 10; do
    echo "Running with ${num_omp_threads} threads..."
    srun --ntasks=1 --nodes=1 --ntasks-per-node=1 "${EXECUTABLE}" --max-iterations 100000 --num-samples 1000000 --dimensions 25 --clusters 50 --spread 3.5 --trials 10 --num_threads ${num_omp_threads}
done