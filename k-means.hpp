#ifndef K_MEANS_HPP
#define K_MEANS_HPP

#include "../rng.h"
#include "helper.hpp"
#include "k-means-iteration.hpp"

#include <vector>


struct KMeans {

	std::vector<std::vector<float>> centroids;
	std::vector<std::vector<float>> points;
	int k;

	int N, D;

	KMeans(int k) : k(k) {

	}


	// k-means++ (slow!)
	std::vector<std::vector<float>> initializeCentroids() {

		std::vector<std::vector<float>> newCentroids;
		std::vector<float> distances(N, 0.0f);

		newCentroids.push_back(rng::choice(points));

		for (int numCentroids = 1; numCentroids < k; ++numCentroids) {

			for (size_t i = 0; i < N; ++i) {
				distances[i] = squaredEuclideanDistance(&newCentroids[0][0], &points[i][0], D);

				for (size_t j = 1; j < numCentroids; ++j) {
					distances[i] = std::min(distances[i], squaredEuclideanDistance(newCentroids[j].data(), points[i].data(), D));
				}
			}

			newCentroids.push_back(points[rng::sample(distances)]);
		}

		return newCentroids;
	}

	void initializeCentroids(const std::vector<std::vector<float>>& pts) {
		N = pts.size();
		D = pts[0].size();
		points = pts;

		centroids = initializeCentroids();
	}


	template <class Iterator = BasicLloydIteration, typename = std::enable_if_t<std::is_base_of_v<LloydIteration, Iterator>>>
	int fit(int maxIter = 500) {

		Iterator iterator(points);

		int iter = 0;
		while (++iter <= maxIter && !iterator.iterate(centroids)) {}

		--iter; // last iteration didn't count (centroids didn't move)

		cout << "Converged in " << iter << " iterations.\n";
		return iter;
	}


	// returns index of closest centroid
	int classify(const std::vector<float>& point) {
		float minDst = 1e30;
		int centroidIndex = -1;

		for (size_t i = 0; i < k; ++i) {
			float dst = squaredEuclideanDistance(point.data(), centroids[i].data(), D);
			if (dst < minDst) {
				minDst = dst;
				centroidIndex = i;
			}
		}

		return centroidIndex;
	}

};



#endif