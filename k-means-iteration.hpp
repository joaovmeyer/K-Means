#ifndef K_MEANS_ITERATION_HPP
#define K_MEANS_ITERATION_HPP

#include "helper.hpp"

#include <vector>

struct LloydIteration {

	const std::vector<std::vector<float>>& points;
	const int N;

	LloydIteration(const std::vector<std::vector<float>>& pts) : points(pts), N(pts.size()) {}

	bool updateCentroids(std::vector<std::vector<float>>& centroids, const std::vector<std::vector<float>>& newCenters, const std::vector<float>& counts) {

		const int K = centroids.size();
		const int D = centroids[0].size();

		bool converged = true;

		for (int i = 0; i < centroids.size(); ++i) {
			if (!counts[i]) {
				converged = converged && (centroids[i] == std::vector<float>(D, 0.0f));
				continue;
			}

			float invCount = 1.0f / counts[i];
			for (int j = 0; j < centroids[0].size(); ++j) {
				float newVal = newCenters[i][j] * invCount;

				converged = converged && (newVal == centroids[i][j]);

				centroids[i][j] = newVal;
			}
		}

		return converged;
	}

	virtual bool iterate(std::vector<std::vector<float>>& centroids) = 0;
};


struct BasicLloydIteration : LloydIteration {

    BasicLloydIteration(const std::vector<std::vector<float>>& pts) : LloydIteration(pts) {}

	bool iterate(std::vector<std::vector<float>>& centroids) override {

		const int k = centroids.size();
		const int D = centroids[0].size();

		std::vector<std::vector<float>> newCenters(k, std::vector<float>(D, 0.0f));
		std::vector<float> counts(k, 0.0f);

		for (size_t i = 0; i < N; ++i) {

			float minDst = 1e30;
			int centroidIndex = -1;

			// find nearest centroid for this point
			for (size_t j = 0; j < k; ++j) {
				float dst = squaredEuclideanDistance(points[i].data(), centroids[j].data(), D);
				if (dst < minDst) {
					minDst = dst;
					centroidIndex = j;
				}
			}

			// accumulate points assigned to given centroid to later get their mean
			counts[centroidIndex]++;
			for (int j = 0; j < D; ++j) {
				newCenters[centroidIndex][j] += points[i][j];
			}
		}

		return updateCentroids(centroids, newCenters, counts);
	}
};













#endif