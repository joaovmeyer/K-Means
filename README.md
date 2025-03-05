# Optimized K-Means

This project implements an **optimized version of the K-Means algorithm**. The goal is to **compare performance** when using techniques such as data parallelism with OpenMP and SIMD, with the objective of **testing the real benefit** of [more complex methods](https://cs.baylor.edu/~hamerly/papers/2014_pca_chapter_hamerly_drake.pdf), such as [filtering with kd-trees](https://www.cs.umd.edu/~mount/Projects/KMeans/pami02.pdf) or triangle inequality based methods like Elkan's, which are planned for future implementation.

## Background

The K-Means algorithm creates K clusters, defined by its centroid. A given point is considered a member of cluster i if its distance to i-th centroid is smaller than its distance to any other centroid. Multiple distance metrics can be used, but in this project **only the Euclidean distance was considered**.

The classic algorithm takes N points and K initial clusters and iteratively calculates all N * K pairwise distances between points and centroids. Then, each point is assigned to the cluster with closest centroid. After each point is assigned a cluster, the clusters are updated such that their centroids are the mean of all points assigned to them. Convergence is reached when the centroids positions remain unchanged compared to the last iteration.

A popular use for the K-Means algorithm is **color quantization**. A general image has thousands of different colors, and for a number of reasons it can be desirable to limit it to only a few. Making each color a vector and each pixel a data point, we can perform the K-Means algorithm to limit the number of colors in an image to the number of clusters used. On this project, **only color quantization was used to test implementations**.

## Optimizations

This project optimizes K-Means **without modifying the classic algorithm**. Instead, it focuses on **efficient resource utilization** to improve performance. The two primary techniques used are:

 - **Multiprocessing**: Distributing computations across multiple CPU cores.
 - **SIMD (Single Instruction, Multiple Data)**: Leveraging vectorized operations to process multiple points simultaneously.
 
These optimizations significantly reduce computation time while maintaining the correctness of the algorithm.

### Multiprocessing (OpenMP)
A straightforward way to accelerate K-Means is multiprocessing. Since each of the N data points is processed independently when assigning clusters, we can **split them into groups** and process each group in parallel. This allows centroid assignments to be **computed concurrently**, and at the end, we just need to update the clusters

The efficiency of this approach **scales with the number of CPU cores**, meaning that a machine with more cores will generally experience a greater speedup. In theory, the speedup should be close to **linear with the number of cores**, making multiprocessing a **reliable method for improving performance**.

### SIMD

SIMD allow us to perform an operation multiple times simultaneously. For example, most computers nowadays have **AVX (Advanced Vector Extensions)**, with 256-bit vector registers that can fit up to 8 single precision float values. This means that one can multiply, add, or even more complex operations on **8 values in a single instruction**.

This project uses AVX vectors to calculate distances between a data point to [multiple centroids at a time](https://jacco.ompf2.com/2020/05/12/opt3simd-part-1-of-2/). Not only that, but we can also then find the closest centroid by comparing [multiple distances simultaneously](https://en.algorithmica.org/hpc/algorithms/argmin/). This technique resulted in significant speedups, especially when working with a large number of clusters. 

Unfortunately, this approach has a few limitations:

 - **Few clusters**: With few clusters, finding the closest centroid is already fast, and the overhead of using SIMD can outweigh its benefits.
 - **High-dimensional data**: As dimensionality increases, distance computations (likely already auto-vectorized) become the bottleneck, and the argmin step gets overwhelmed by the expensive distances calculations anyways.

Overall, **this SIMD-based approach is highly effective**, especially in **low-dimensional datasets with many clusters**, where the benefits are most pronounced.

## Benchmarks

Some notes on the benchmarks:

 - **Centroid initialization**: On all benchmarks, **centroid initialization was random** (since my crude implementation of k-means++ was too slow and I wanted to test big values of K).
 - **What's measured?** Benchmarks only include the iterative process of doing Lloyd's iteration until convergence or a **maximum of 50 iterations**, and things like **centroid initialization were not included**.
 - **How it's measured?** Each benchmark code was executed **4 times** and the **mean of the results were used**.
 - **Speedup calculation**: I measured **average time per iteration** since other implementations often required a different number of iterations to converge. To ensure a fair comparison, I used the **same initial centroids across implementations**.
 - **Cores used**: Unless clearly stated, all multiprocessing code was executed on 4 cores.
 - **Notation**: N = number of data points, D = number of dimensions, K = number of clusters.

All the c++ code was compiled with GCC 13.2.0 using the flags: `-std=c++17`, `-fopenmp`, `-march=native`, `-O2`.
<br>
The scikit-learn version used was 1.6.1.

---

**Using image nature.jpg**: N = 756000, D = 3.
| K = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256 | K = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64 |
|-|-|
| ![speedup_all](https://github.com/user-attachments/assets/7cf8fddf-2198-453c-8e06-2d8e6a825d80) | ![speedup_64](https://github.com/user-attachments/assets/c6d7db28-76d6-4ce2-a199-1f22adcdc562) |
| ![tpi_all](https://github.com/user-attachments/assets/a21f90f3-3052-4c92-9f1c-979b6596737c) | ![tpi_64](https://github.com/user-attachments/assets/6362815d-464e-47df-8e16-c1e509368d4e)
 |

It's weird to me how close the graphs for SIMD and scikit-learn are. The natural thing to think is that scikit-learn is not using multiprocessing and just doing something similar to what I'm doing, but I don't believe that's it, because my CPU usage pretty clearly goes to 100% when running the scikit-learn code, so it must be using all cores. I really think it's just a coincidence.

---

**Using image monkey.jpg**: N = 262144, D = 3, K = 16.
| Time Per Iteration (ms) | Speedup |
|-|-|
| ![tpi_num_threads](https://github.com/user-attachments/assets/2ebe62e7-ad84-4bd6-8f25-c60210d544cf) | ![speedup_num_threads](https://github.com/user-attachments/assets/0801e1e6-df18-48d5-9830-a6522a304979) |

This shows that scikit-learn was indeed using **4 cores** before, and my implementation using SIMD can get similar results using **only 1 core**. When both are using the same number of cores, my implementation achieves **speedups of around 3x** in this example with K = 16. It's also nice to see that speedups grow linearly with the number of cores, as we expected.

## Disclaimers

This code is not even close to production-quality. The **only** optimized part is the centroid assignment and points accumulation, all the rest was left **unchanged from a basic implementation**. Overall, the objective of this project is **not** providing a highly efficient implementation of the K-Means algorithm as a whole, but mainly **comparing techniques** of doing Lloyd's iteration.

Color quantization was chosen because it fits nicely the filtering algorithm using kd-trees (that I plan to add here soon), as they suffer from the curse of dimensionality and color quantization has 3 dimensions (in general). It turns out it also favors my SIMD approach. **Expect benchmarks on higher dimensional data to be not as positive**.

Some inconsistencies can be found when using multiprocessing due to **different order of floating-point operations**, leading to slightly **different results each time** because of numerical instability, but this is **not a problem in general**.

I don't know how fair it is to compare my results with scikit-learn, as not only it's a big library that has to worry about way more things than I did, but it's also in Python. I have close to no understanding on how to optimize a Python code, and it might be impossible to get similar results with the same techniques as I did, but still, 3x worse results **seems unjustified**, but I could be wrong.
