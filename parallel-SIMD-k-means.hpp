#pragma once

#include "k-means-iteration.hpp"
#include <omp.h>


struct ParallelSIMDLloydIteration : LloydIteration {

	ParallelSIMDLloydIteration(const std::vector<std::vector<float>>& pts) : LloydIteration(pts) {}

	bool iterate(std::vector<std::vector<float>>& centroids) override {

		const int k = centroids.size();
		const int D = centroids[0].size();

		std::vector<std::vector<float>> newCenters(k, std::vector<float>(D, 0.0f));
		std::vector<float> counts(k, 0.0f);


		// round up and add padding with std::vector<float>(D, inf) so they are never closest to any point
		int numVecs = (k + 7) / 8;

		// prepare all the vectors we will be using (not exactly efficient but only done once per iteration)
		// permuting the vectors in such a way that index1 < index2 <=> permutedIndex1 % 8 < permutedIndex2 % 8
		__m256* vecs = allocAVX(D * numVecs);

		// fill vecs with infs
		for (int i = 0; i < D * numVecs; ++i) vecs[i] = _mm256_set1_ps(std::numeric_limits<float>::infinity());

		for (int j = 0; j < k; ++j) {
			int y = j % numVecs;
			int x = j / numVecs;

			for (int d = 0; d < D; ++d) {
				setValue(vecs[y * D + d], centroids[j][d], x);
			}
		}


		#pragma omp parallel
		{

			std::vector<std::vector<float>> newCentersThread(k, std::vector<float>(D, 0.0f));
			std::vector<float> countsThread(k, 0.0f);

			#pragma omp for nowait
			for (int i = 0; i < N; ++i) {

				const __m256 increment = _mm256_set1_ps(1.0f);
				__m256 minIdxs = _mm256_mul_ps(_mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f), _mm256_set1_ps((float) numVecs));
				
				// first 8 distances can be calculated here to avoid an unnecessary calls to updateArgmin
				// this is actually important when K is small, and that's quite common in k-means
				__m256 minDistances = calculateDistances8x(points[i].data(), &vecs[0], D);

				// indexes of next clusters we will work on
				__m256 currIdx = _mm256_add_ps(minIdxs, increment);


				for (int j = 1; j < numVecs; ++j) {

					__m256 dst = calculateDistances8x(points[i].data(), &vecs[j * D], D);

					// update argmin and min distance in every slice
					updateArgmin(minIdxs, currIdx, minDistances, dst);

					currIdx = _mm256_add_ps(currIdx, increment);
				}

				// pass the contents of minIdxs to an array
				float idxsArr[8];
				_mm256_storeu_ps(idxsArr, minIdxs);

				// this creates a problem: argminAVX will return the index of the first element in minDistances that is equal to the min 
				// value in minDistances, but because the real indices are in idxsArr, this will NOT be the real argmin. It will actually
				// be the argmin of the indices % 8 (so for example, if centroids 6 and 17 are equidistant to the point, this will give
				// centroid 17 % 8 = 1 as argmin instead of 6 % 8 = 6). This is not actually a problem because the k-means algorithm doesn't
				// specify to which cluster a point should be assigned to if two or more share the same (and minimum) distance, so we could
				// choose any, but just so this is equal to the scalar version, I "fixed" it with the permutations done above
				int centroidIndex = (int) idxsArr[argminAVX(minDistances)];

				// accumulate points assigned to given centroid to later get their mean
				countsThread[centroidIndex]++;
				for (int j = 0; j < D; ++j) {
					newCentersThread[centroidIndex][j] += points[i][j];
				}
			}

			// accumulate results from all threads
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

		freeAVX(vecs);

		return updateCentroids(centroids, newCenters, counts);
	}
};
