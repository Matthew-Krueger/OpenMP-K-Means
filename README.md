---
title: OpenMP - K-Means Clustering
author: Matthew Krueger <contact@matthewkrueger.com>
description: An implementation of K-Means Clustering written in C++23 using OpenMP.
slug: 20251205-openmp-kmeans
date: 2025-12-05
categories:
- Portfolio
- OpenMP
- Clustering
- K-Means
- Parallel Computing
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

## 0. GitHub Repository
The repository for this project can be found [here](https://github.com/Matthew-Krueger/OpenMP-K-Means).

## 1. Summary
This report describes the implementation and parallelization of K-Means Clustering using OpenMP,
in comparison to serial execution.
This is comparing and contrasting focusing on OpenMP, and building on the prior work done in MPI.
You can read the prior report at [this link](https://www.matthewkrueger.com/blog/20251023-kmeans-mpi).
Only relevant changes over MPI are described here, except for basic overview.

### 1.1 Algorithm Overview (review)
K-Means Clustering is a popular clustering algorithm widely used in machine learning.
It is a method of unsupervised learning that aims to partition a set of data into a number of clusters,
where each data point is assigned to the cluster with the closest average (mean).
In effect, this algorithm finds groups of data points that are similar to each other.
In an iteration loop, the algorithm first finds to which previous cluster any given data point belongs.
From there, it calculates the average of all data points in *that cluster*, and this average is now the new "centroid."
This process is then repeated until convergence, or until a maximum number of iterations is reached.
Convergence is when the centroids are not changing significantly between iterations,
demarcated by a specified tolerance (epsilon).

---

## 2. Explanation of Design

### 2.1. Algorithm Choice and Serial Implementation
See the MPI report for the baseline serial design, which remains unaltered here.
(Note hungarian algorithm was added, which will be referenced later. It does not impact timing and thus is not included here)

### 2.2. Parallelization Strategy

The algorithm was parallelized via OpenMP.
Specifically, only one part of the serial algorithm was changed.
First, because OpenMP does not natively support transform operations from the standard library such as `std::transform`,
I instead opted to use a regular for loop to iterate over the points in the dataset.
The local data was handled very similarly to MPI, for this reason alone: in my testing, a custom reduction function
did not really work as intended and caused a variety of issues despite officially being supported.
Instead, I created a global accumulator and a local accumulator for each thread.
The local accumulators are used to accumulate the sum of each thread's individual work unit,
and then each thread writes its local accumulator to the global accumulator.

This follows this order:
```c++
            // create a global accumulator
            std::vector<Point> globalAccumulators(
                m_PreviousCentroids.size(),
                Point(std::vector<double>(m_DataSet[0].numDimensions(), 0.0), 0)
            );

            // now that we have that, we can now accumulate
            // since OMP does not support std::accumulate in the manner needed here, and this pattern is a bit complicated, I am going to
            // use critical in this case
#pragma omp parallel
            {

                // set up the thread local accumulators
                std::vector<Point> localAccumulators(
                    m_PreviousCentroids.size(),
                    Point(std::vector<double>(m_DataSet[0].numDimensions(), 0.0), 0)
                );

                // use nowait so threads can continue when they want
#pragma omp for schedule(static) nowait
                for (size_t m_DataSetPointIndex = 0; m_DataSetPointIndex < m_DataSet.size(); ++m_DataSetPointIndex) {
                    if (auto it = m_DataSet[m_DataSetPointIndex].findClosestPointInVector(m_PreviousCentroids); it != m_PreviousCentroids.end()) {
                        const size_t idx = std::distance(m_PreviousCentroids.begin(), it);

                        // Accumulate into LOCAL vector. No locks needed.
                        localAccumulators[idx] += m_DataSet[m_DataSetPointIndex];
                        localAccumulators[idx].setCount(localAccumulators[idx].getCount() + 1);
                    }
                }

                // lock the global accumulators
                // and then allow each thread to do their thing
                // in this case this particular antipattern is easier than using the custom reductor function
#pragma omp critical
                {
                    for (size_t k = 0; k < globalAccumulators.size(); ++k) {
                        globalAccumulators[k] += localAccumulators[k];
                        globalAccumulators[k].setCount(globalAccumulators[k].getCount() + localAccumulators[k].getCount());
                    }
                }

            } // end omp parallel
```

The only other change from MPI is that I had to rewrite the main function to remove the MPI calls.
In here, I set the `omp_set_num_threads` to the number of threads passed by the command line switch.

The first pragma is *specifically* to do the parallel accumulation.
Nowait was chosen so the threads can continue opportunistically and enter the critical at any time,
rather than synchronizing and then entering one at a time.
Due to the omp critical after, there will be no race conditions other than as described in Section 2.3,
which reduces overhead and barrier costs.

### 2.3 Considerations for Parallelization

(largely unchanged from MPI)

One additional consideration is that since we are still using double (IEEE 754) precision floating point numbers,
due to the non-deterministic order of thread completion compared to MPI which uses deterministic ordering for Allreduce,
we may lose some precision in this reduction and have some small variance.
Largely this is due to IEEE 754's non-associativity of addition, and thus the order of operations.
It does not affect us in MPI as the order of operations is deterministic leading to the same error over time.

There is still a possibility of contention over the global accumulator, especially with larger K values.
However, in this test I only used a K of 50, and the operation is fundamentally O(n) making parallelization largely
not worthwhile in this context.

---

## 3. Benchmarking Details

### 3.1. Experiment Setup

(largely unchanged from MPI, repeated for clarity)
I ran my experiment on the OSC (Ohio Supercomputing Center) Cardinal Cluster.
Details of the hardware can be found [on their website](https://www.osc.edu/resources/technical_support/supercomputers/cardinal).
I did not request any specific hardware, only 10 threads on one node. I did not specify NUMA domains.
I used `openmpi/5.0.2`, `boost/1.83.0`, `cmake/3.25.2`, and `gcc/13.2.0`.
It should be noted that `gcc/13.2.0` only supports the preview version of C++23, and no other compiler supported
specific features I used, specifically zip views.
OpenMPI is not linked unless specified. It was not specified for any test runs, but the option remains in the code.

I ran each test a total of 10 times and reported the runtime of each run.
`srun` was invoked once per number of processes,
and the number of threads was specified as a command line argument and the `OMP_NUM_THREADS` environment variable.
The script used to run the experiments is available in the root directory of this repository, called `benchmark.slurm`.

I regenerated the dataset from the same seed for each invocation of srun, but I did not regenerate the dataset for each
set of batches. This did not affect my result as I only timed the actual run of the K-Means algorithm, NOT of dataset
generation.
(it is worthwhile to mention that due to HBMe memory and other factors on cardinal,
there may be some slight memory training that occurs as more runs are performed, but the impact is likely negligible)

Each dataset was generated using the following parameters:
*   Number of samples: 1,000,000
*   Number of dimensions: 25
*   Number of clusters (K): 50
*   Dataset generation spread/characteristics: 3.5
*   Random seed used for reproducibility: 1234
*   Max iterations: 100,000
*   Default convergence threshold: 0.0001

Because of my timer, it used std::chrono::high_resolution_clock to measure the time elapsed.
10 Trials were executed for each configuration (process count).

All statistical analysis was done on Apple Numbers, and is included in the `/Results/omp` directory.
MPI Results are included in the `/Results/mpi` directory for reference, though for any corrections or clarifications,
please refer to the MPI writeup referenced at the top of this document.

### 3.2. Benchmark Results

The results of the run are as follows:

#### 3.2.1. Results for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10

OpenMP Results:
##### Table of Values for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10
| Number Threads | Number Samples (Average) | Number Dimensions (Average) | Number Clusters (Average) | Spread (Average) | Seed (Average) | Wall Time (s) (Average) | Wall Time (s) (STDEV) | Iteration Count (Average) |
|----------------|--------------------------|-----------------------------|---------------------------|------------------|----------------|-------------------------|-----------------------|---------------------------|
| 1              | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 148.4354                | 1.003                 | 250                       | 1.000 | 100.000% | |
| 2              | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 74.4632                 | 0.368                 | 250                       | 1.993 | 99.670% | 0.003 |
| 3              | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 49.47206                | 0.137                 | 250                       | 3.000 | 100.013% | -0.000 |
| 4              | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 37.31448                | 0.055                 | 250                       | 3.978 | 99.449% | 0.002 |
| 5              | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 29.83018                | 0.088                 | 250                       | 4.976 | 99.520% | 0.001 |
| 6              | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 24.83431                | 0.057                 | 250                       | 5.977 | 99.617% | 0.001 |
| 7              | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 21.3100                 | 0.053                 | 250                       | 6.966 | 99.507% | 0.001 |
| 8              | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 18.7791                 | 0.065                 | 250                       | 7.904 | 98.803% | 0.002 |
| 9              | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 16.59767                | 0.083                 | 250                       | 8.943 | 99.368% | 0.001 |
| 10             | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 14.9674                 | 0.067                 | 250                       | 9.917 | 99.173% | 0.001 |

#### Speedup for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10
![Speedup Plot](/Results/openmp/Speedup.png)

#### Efficiency for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10
![Efficiency Plot](/Results/openmp/Efficiency.png)

#### Karp-Flatt for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10
![Karp-Flatt Plot](/Results/openmp/Karp-Flatt.png)

#### Run Time (Seconds) for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10
![Run Time Plot](/Results/openmp/Runtime-seconds.png)

#### Run Time (Seconds, STDEV) for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10
![Run Time Plot](/Results/openmp/Runtime-seconds-sigma.png)

### 3.3. Interpretation of Results

#### 3.3.1. Standalone Interpretation
My interpretations of the results are largely the same as they were under MPI.

The speedup is very near linear, and the efficiency is very close to linear.
At the problem size chosen, the implementation displays near optimal scaling.

The data shows no real sign of a scaling limit approaching at this point in time as p increases,
but there are some inherent scaling limits that could show, especially at higher K.

#### 3.3.2. Interpretation in Context with MPI
The MPI results are very similar to the OpenMP results.

For ease of comparison, here are the results of the MPI implementation (GitHub link at the top of this document):

##### Table of Values for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10
| Number Processes | Number Samples (Average) | Number Dimensions (Average) | Number Clusters (Average) | Spread (Average) | Seed (Average) | Wall Time (s) (Average) | Wall Time (s) (STDEV) | Iteration Count (Average) |
|------------------|--------------------------|-----------------------------|---------------------------|------------------|----------------|-------------------------|-----------------------|---------------------------|
| 1                | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 147.6953                | 1.16059               | 250                       |
| 2                | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 75.18676                | 0.22864               | 250                       |
| 3                | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 50.24906                | 0.23718               | 250                       |
| 4                | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 37.81581                | 0.12123               | 250                       |
| 5                | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 30.18961                | 0.11872               | 250                       |
| 6                | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 25.20847                | 0.06285               | 250                       |
| 7                | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 21.6393                 | 0.04091               | 250                       |
| 8                | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 18.98175                | 0.03637               | 250                       |
| 9                | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 16.8384                 | 0.04975               | 250                       |
| 10               | 1000000                  | 25                          | 50                        | 3.5              | 1234           | 15.1942                 | 0.03572               | 250                       |


##### Speedup for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10
![Speedup Plot](/Results/mpi/Speedup.png)

##### Efficiency for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10
![Efficiency Plot](/Results/mpi/Efficiency.png)

##### Karp-Flatt for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10
![Karp-Flatt Plot](/Results/mpi/Karp-Flatt.png)

##### Run Time (Seconds) for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10
![Run Time Plot](/Results/mpi/Runtime-seconds.png)

##### Run Time (Seconds, STDEV) for 1,000,000 Samples, 25 Dimensions, 50 Clusters, Process Count 1-10
![Run Time Plot](/Results/mpi/Runtime-seconds-sigma.png)

The MPI results are very similar to the OpenMP results.
However, there are four notable differences:
1. The MPI implementation is slower than the OpenMP implementation.
2. The MPI implementation is less efficient than the OpenMP implementation.
3. The MPI implementation has a higher karp-flatt coefficient than the OpenMP implementation.
4. The OpenMP implementation has a higher karp-flatt coefficient "jitter." (that is, the karp flatt is less consistent when adding more threads)

In large part these are almost certainly due to MPI communication overhead.
Assuming the MPI implementation was run on only one node,
the extra overhead would primarily come to serialization and deserialization of data.
Put together, the differences are mostly negligible.
However, in a larger setup running thousands of times to find the optimal solution,
not just a solution, the differences could become more pronounced.
At 250 iterations across all runs, that is 250 specific time-sensitive serializations/deserialization in each AllReduce.
OpenMP, being a single process, largely avoids this memory, compute, and network traffic overhead.
OpenMP's implementation does face another problem, however,
which is the aforementioned associativity of IEEE 754 doubles.
There also may be some additional jitter if the scheduler put the job between NUMA domains since it was not specified
in the slurm script.
However, the data does not really reflect any scaling problems commonly associated with NUMA communication,
and can be called essentially "good enough."

Both implementations also received a statistically significant R^2 value of 1 when describing the best fit line
of their wall times.
This shows that their performance is consistent.

However, the OpenMP implementation did have a standard deviation value around double that of the MPI implementation.
I did not check system utilization at the time, however, this system is very heavily used by GPU jobs and is normally
under quite immense memory pressure because of those jobs.
If there was any significant process thrashing, it could cause variability in the results from run to run or from day to day.
In addition, since we did run on the backfill queue, there is a possibility that the specific system cores allocated
may not have been 100% stable either, which could also cause variability.
Lastly, it could also just be variable due to shared memory access patterns,
leading to potentially different memory access patterns between the two implementations.
This could also be particularly notable due to the presence of HBMe memory on these machines.
*Especially* on the backfill queue, and especially with each thread having access to (even if it is not reading) the
whole dataset at once.

---

## 4. Verification

### 4.1. Correctness Strategy

I added an implementation of the Hungarian Algorithm to address label switching and to give a squared distance metric.
This gives the "energy state of the system" which we can use to imperially verify the correctness of the algorithm.

When running the algorithm on a reduced subset of the dataset, I noticed that I was in a very high energy state.
The algorithm was converging, but several centroids were converging to the same location rather than moving
towards each group.
Upon closer inspection of the MPI algorithm and comparison with the serial implementation,
I noticed this issue was present in all versions.
This is not *necessarily* an issue, however, it is an indication that the algorithm is converging to a local minimum,
not the intended global minimum.
I did not attempt to fix this issue at this time as the main goal was to get a specific solve of the algorithm working.
The energy state of each solve was not recorded in the runs.

I also validated the algorithm was correctly implemented by running it locally on my laptop where I have more control.
I ran each test once with 10,000 samples, 3 dimensions, 10 clusters, and a spread of 3.5.
The results were as follows:
```terminaloutput
mckrueg@MacBookPro cmake-build-debug % ./kmeans_openmp --max-iterations 100000 --num-samples 10000 --dimensions 3 --clusters 10 --spread 3.5 --trials 1 --num-threads 1 
Known Good Centroids:
        Point: (22668,47979.2,66262.2)
        Point: (38115.9,36405.6,58620.5)
        Point: (48770.2,33666.7,-29679)
        Point: (43568.5,50359.4,-59226.1)
        Point: (26191.8,25562.7,-9607.56)
        Point: (8152.51,27913.7,-47470)
        Point: (54788.4,36943.7,-68192.5)
        Point: (47713.3,50874.8,68768.3)
        Point: (320.971,22865.4,-36587.4)
        Point: (51269.9,41025.3,-44354.5)
Hungarian Time: 9e-06
Total Squared Euclidian Distance: 3.08616e+10
1,10000,3,10,3.5,1234,0.006491,yes,14,0
Calculated Centroids:
        Point: (8152.39,27913.8,-47470)with a count of 1000
        Point: (49599.3,40498.8,-50363)with a count of 4000
        Point: (22670.4,47981.7,66260.1)with a count of 245
        Point: (22668,47981,66265.8)with a count of 281
        Point: (47713.4,50874.8,68768.1)with a count of 1000
        Point: (38116,36405.6,58620.6)with a count of 1000
        Point: (22663.8,47979.4,66260.7)with a count of 234
        Point: (22669.4,47975.7,66262)with a count of 240
        Point: (26191.8,25562.7,-9607.51)with a count of 1000
        Point: (320.922,22865.3,-36587.4)with a count of 1000
mckrueg@MacBookPro cmake-build-debug % ./kmeans_openmp --max-iterations 100000 --num-samples 10000 --dimensions 3 --clusters 10 --spread 3.5 --trials 1 --num-threads 10
Known Good Centroids:
        Point: (22668,47979.2,66262.2)
        Point: (38115.9,36405.6,58620.5)
        Point: (48770.2,33666.7,-29679)
        Point: (43568.5,50359.4,-59226.1)
        Point: (26191.8,25562.7,-9607.56)
        Point: (8152.51,27913.7,-47470)
        Point: (54788.4,36943.7,-68192.5)
        Point: (47713.3,50874.8,68768.3)
        Point: (320.971,22865.4,-36587.4)
        Point: (51269.9,41025.3,-44354.5)
Hungarian Time: 1.8e-05
Total Squared Euclidian Distance: 3.08616e+10
10,10000,3,10,3.5,1234,0.007402,yes,14,0
Calculated Centroids:
        Point: (8152.39,27913.8,-47470)with a count of 1000
        Point: (49599.3,40498.8,-50363)with a count of 4000
        Point: (22670.4,47981.7,66260.1)with a count of 245
        Point: (22668,47981,66265.8)with a count of 281
        Point: (47713.4,50874.8,68768.1)with a count of 1000
        Point: (38116,36405.6,58620.6)with a count of 1000
        Point: (22663.8,47979.4,66260.7)with a count of 234
        Point: (22669.4,47975.7,66262)with a count of 240
        Point: (26191.8,25562.7,-9607.51)with a count of 1000
        Point: (320.922,22865.3,-36587.4)with a count of 1000
mckrueg@MacBookPro cmake-build-debug % 
```

The serial algorithm's converged outputs are essentially identical to the parallel algorithm's converged outputs.
There may be some difference not shown in the default output due to the IEEE 754 double being not associative as well.
NOTE: I ran in 10 core mode to match OSC. Due to P-Core eviction on my MacBook, the speedup is not stable above seven cores.

## 4.2 Future Optimization, Verification Strategy, and Methods

For future development,
an important enhancement would be to incorporate the ability to execute multiple k-means trials concurrently.
Each trial could leverage different initial centroids or random seeds,
allowing for a comprehensive exploration of the solution space.
This approach would facilitate identifying the optimal configuration by comparing the results across various runs,
ultimately aiming to minimize the total energy state of the system (e.g., within-cluster sum of squares).

While the initial focus of this project was primarily on implementing the core solver algorithm,
this extended capability was deferred.
However, it represents a strong candidate for demonstrating effective parallelization,
as independent trials can be run in parallel,
significantly reducing the overall computation time required to achieve a more robust and optimal clustering solution.

---

## 5. Reflection

### 5.1. Learning Experience

I learned that while a problem appears simple on the surface, it may be more complex than you think.
For example, K-Means is a basic algorithm and relatively trivial to parallelize.
However, the specific details of the operations are very detailed and require careful thought and planning.

I also learned that C++23 does make life easier, but it does come at a cost of complexity.
Much of the pipeline code I wrote is substantially longer than it would be if I had done it the old-fashioned way.
However, in the end result, it often yields a more performant result.

I did also learn that the C++23 features are **NOT** a silver bullet capable of solving all problems.
I had several places in my code where the fancy C++23 features
were actually substantially slower than direct C++17 code.
In these cases, I had to remove them and use regular loops instead.

I was able to gain this insight through my profiling system, which gives me detailed timing information for
every level of the call stack.

I also learned that Boost.MPI is a very powerful library, and avoiding manually flattening of points prevents
substantial errors that caused various types of heap corruption.
For instance, I had one bug while attempting to remove the functor (as boost.MPI::all_reduce has some weird performance
implications when using a custom reduction functor), but any time I attempted to copy data into `m_CurrentCentroids`,
I was unable to do so without having a segmentation fault. I was even able to copy the data several times before that,
but as soon as it went into m_CurrentCentroids, it faulted out.

I believe most aspects were done pretty well, but I do wish I had a bit more time to go through and organize the code a
bit better.

### 5.2. Trade-offs

I had to decide not to use a custom reduction functor due to complexity and performance concerns.
It also did not seem to reliably work with my implementation of the Point class, since fundamentally a vector of points
is really just a vector of a vector of doubles.
In effect, `std::vector<Point>` is really just a wrapper over `std::vector<std::vector<double>>`.
This makes custom reduction functors difficult to implement.
While a custom reduction functor remains the best choice,
in the case of high values of `k`,
a multiphasic reduction may be preferable if a custom reduction functor cannot be used or produces unreliable results.

This is in addition to the trade-offs mentioned in the MPI build:
The closest thing was that I had to use the pipeline system to generate the dataset, as not doing so was unsustainable
on my machine due to limited memory.
While this would have been fine on OSC, I really wanted this program to be runnable on my local machine for
ease of development.

### 5.3. Advice for Future Students

My advice remains unchanged from the previous assignment.
I would advise making heavy use of timers and profiling to be able to understand the detailed implications of your code.
I would also heavily advise the heavy use of modern C++ functions, as well as auto vectorization optimizations
to speed up computations to the maximum extent possible.

---

## 6. Deliverables (Checklist)

*   [X] Code (Serial and Parallel versions)
*   [X] Build system (`CMakeLists.txt`)
*   [X] SLURM scripts for OSC
*   [X] Benchmark data (CSV in `/Results` directory)
*   [X] `README.md` (This report, following the specified sections)

### 6.1 AI Use
No significant AI use beyond tab autocomplete and generation of the hungarian algorithm code (which is not required for the project).
There were several discussions over a custom reduction functor, but I eventually scrapped that idea entirely to reduce the surface area of the shared memory contention.
As such, no transcripts are included in this submission.

---