#pragma once

#include "k-means-iteration.hpp"



// one of the biggest problems I faced was trying to make this efficient even for a small K.
// something as simple as changing to SSE when K is small (<= 4 is definitely enough, but maybe even bigger K) is enough
// to give somewhat of a performance boost. Another idea that seemed to work well for even not so small K is using AVX
// to calculate the distance of one centroid to 8 points, doing it 4 times, then using SSE just for the argmin. I did
// test something like this and results were promising, but the code was too nasty and big so I won't even bother
// another problem is when dimensions get big: I didn't test this situation much, but an obvious impact is that
// the code will take MORE time to execute while the time saved by using SIMD will stay the SAME, so the speedup,
// measured with relative performance, will start to decrease. Maybe this SIMD version will even get slower because
// the way I calculate distances here probably is not optimal at high dimensions, but I didn't test it to be sure


struct SIMDLloydIteration : LloydIteration {

	SIMDLloydIteration(const std::vector<std::vector<float>>& pts) : LloydIteration(pts) {}

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
			uint32_t y = j % numVecs;
			uint32_t x = j / numVecs;

			for (int d = 0; d < D; ++d) {
				setValue(vecs[y * D + d], centroids[j][d], x);
			}
		}


		for (int i = 0; i < N; ++i) {

			const __m256 increment = _mm256_set1_ps(1.0f);
			__m256 minIdxs = _mm256_mul_ps(_mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f), _mm256_set1_ps((float) numVecs));
			
			// first 8 distances can be calculated here to avoid an unnecessary cmp, blendv and min
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
			counts[centroidIndex]++;
			for (int j = 0; j < D; ++j) {
				newCenters[centroidIndex][j] += points[i][j];
			}
		}

		freeAVX(vecs);

		return updateCentroids(centroids, newCenters, counts);
	}
};
