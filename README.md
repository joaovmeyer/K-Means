# Optimized K-Means

This project implements an **optimized version of the K-Means algorithm**. The goal is to **compare performance** when using techniques such as data parallelism with OpenMP and SIMD, with the objective of **testing the real benefit** of more complex methods, such as filtering with kd-trees[1](https://www.cs.umd.edu/~mount/Projects/KMeans/pami02.pdf) or triangle inequality based methods like Elkan's, which are planned for future implementation.

## Background

The K-Means algorithm creates K clusters, defined by its centroid. A given point is considered a member of cluster i if its distance to i-th centroid is smaller than its distance to any other centroid. Multiple distance metrics can be used, but in this project only the Euclidean distance was considered.

The classic algorithm takes N points and K initial clusters and iteratively calculates all N * K pairwise distances between points and centroids. Then, each point is assigned to the cluster with closest centroid. After each point is assigned a cluster, the clusters are updated such that their centroids are the mean of all points assigned to them. Convergence is reached when the centroids positions remain unchanged compared to the last iteration.

A popular use for the K-Means algorithm is color quantization. A general image has thousands of different colors, and for a number of reasons it can be desirable to limit it to only a few. Making each color a vector and each pixel a data point, we can perform the K-Means algorithm to limit the number of colors in an image to the number of clusters used. On this project, only color quantization was used to test implementations.

## Optimizations

This project optimizes K-Means **without modifying the classic algorithm**. Instead, it focuses on **efficient resource utilization** to improve performance. The two primary techniques used are:

 - **Multiprocessing**: Distributing computations across multiple CPU cores.
 - **SIMD (Single Instruction, Multiple Data)**: Leveraging vectorized operations to process multiple points simultaneously.
 
These optimizations significantly reduce computation time while maintaining the correctness of the algorithm.

### Multiprocessing
A straightforward way to accelerate K-Means is **multiprocessing**. Since each of the N data points is processed independently when assigning clusters, we can split them into groups and process each group in parallel. This allows centroid assignments to be computed concurrently, and at the end, we just need to update the clusters

The efficiency of this approach **scales with the number of CPU cores**, meaning that a machine with more cores will generally experience a greater speedup. In theory, the speedup should be close to linear with the number of cores, making multiprocessing a reliable method for improving performance.

### SIMD

SIMD allow us to perform an operation multiple times simultaneously. For example, most computers nowadays have **AVX (Advanced Vector Extensions)**, with 256-bit vector registers that can fit up to 8 single precision float values. This means that one can multiply, add, or even more complex operations on **8 values in a single instruction**.

This project uses AVX vectors to calculate distances between a data point to **8 centroids at a time**[2](https://jacco.ompf2.com/2020/05/12/opt3simd-part-1-of-2/). Not only that, but we can also then find the closest centroid by comparing **8 distances simultaneously**[3](https://en.algorithmica.org/hpc/algorithms/argmin/). This technique resulted in significant speedups, especially when working with a large number of clusters. 

unfortunately, this approach has a few limitations:

 - **Few clusters**: With few clusters, finding the closest centroid is already fast, and the overhead of using SIMD can outweigh its benefits.
 - **High-dimensional data**: As dimensionality increases, distance computations become the bottleneck, and the argmin step gets overwhelmed by the expensive distances calculations anyways.

Overall, **this SIMD-based approach is highly effective**, especially in **low-dimensional datasets with many clusters**, where the benefits are most pronounced.

## Disclaimers

This code is not even close to production-quality. The **only** optimized part is the centroid assignment and points accumulation, all the rest was left **unchanged from a basic implementation**. Overall, the objective of this project is **not** providing a highly efficient implementation of the K-Means algorithm as a whole, but mainly **comparing techniques** of doing Lloyd's iteration. Color quantization was chosen because it fits nicely the filtering algorithm using kd-trees (that I plan to add here soon), as they suffer from the curse of dimensionality and color quantization has 3 dimensions (in general). It turns out it also favors my SIMD approach. **Expect benchmarks on higher dimensional data to be not as positive**. 

Some notes on the benchmarks:

 - **Centroid initialization**: On all benchmarks, **centroid initialization was random** (since my crude implementation of k-means++ was too slow and I wanted to test big values of K).
 - **What's measured?** Benchmarks only include the iterative process of doing Lloyd's iteration until convergence, and things like **centroid initialization were not included**.
 - **Speedup calculation**: I measured **average time per iteration** since other implementations often required a different number of iterations to converge. To ensure a fair comparison, I used the **same initial centroids across implementations**.

 Also, some inconsistencies can be found when using multiprocessing:

 - **Non-integer data**: Floating-point precision issues can lead to slight inconsistencies in results due to different orders of operations.
 - **Big values**: When using large images, accumulating the points would also lead to loss of precision if values got too large, causing a similar issue
