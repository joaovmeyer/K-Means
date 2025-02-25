#pragma once

#include "k-means-iteration.hpp"
#include <omp.h>


// I made a really poor choice of words. Every time I use parallel, I really mean
// just multiprocessing. I will always refer to SIMD parallelism as *just* SIMD

struct ParallelLloydIteration : LloydIteration {

    ParallelLloydIteration(const std::vector<std::vector<float>>& pts) : LloydIteration(pts) {}

	bool iterate(std::vector<std::vector<float>>& centroids) override {

		const int k = centroids.size();
		const int D = centroids[0].size();

		std::vector<std::vector<float>> newCenters(k, std::vector<float>(D, 0.0f));
		std::vector<float> counts(k, 0.0f);

		#pragma omp parallel
		{

			// create vectors to accumulate the points for each thread
			std::vector<std::vector<float>> newCentersThread(k, std::vector<float>(D, 0.0f));
			std::vector<float> countsThread(k, 0.0f);

			#pragma omp for nowait
			for (size_t i = 0; i < N; ++i) {

				float minDst = 1e30;
				int centroidIndex = -1;

				for (size_t j = 0; j < k; ++j) {
					float dst = squaredEuclideanDistance(points[i].data(), centroids[j].data(), D);
					if (dst < minDst) {
						minDst = dst;
						centroidIndex = j;
					}
				}

				countsThread[centroidIndex]++;
				for (int j = 0; j < D; ++j) {
					newCentersThread[centroidIndex][j] += points[i][j];
				}
			}

			// combine the results of all threads
			#pragma omp critical
			{
				for (int j = 0; j < k; ++j) {
					counts[j] += countsThread[j];

					for (int l = 0; l < D; ++l) {
						newCenters[j][l] += newCentersThread[j][l];
					}
				}
			}

		}

		return updateCentroids(centroids, newCenters, counts);
	}
};