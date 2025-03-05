#include <vector>
#include <iostream>
#include <string>

#include "../rng.h"

#include "k-means.hpp"
#include "parallel-k-means.hpp"
#include "SIMD-k-means.hpp"
#include "parallel-SIMD-k-means.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "STB/stb_image.h"

using namespace std;




class Timer {
public:
    Timer() {
        start();
    }

    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
    }

    double elapsedSeconds() const {
        return std::chrono::duration<double>(end_time_ - start_time_).count();
    }

    double elapsedMilliseconds() const {
        return std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
    }

    void displayTime() const {
        std::cout << "Elapsed time: " << elapsedMilliseconds() << " ms\n";
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};






template <typename T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& m) {

    os << "[";

    for (size_t j = 0; j < m.size(); ++j) {
        os << m[j];

        if (j + 1 < m.size()) {
            os << ", ";
        }
    }

    os << "]";

    return os;
}


std::vector<std::vector<float>> getImageData(const string& src) {

    int w, h, n;
    unsigned char *img = stbi_load(src.c_str(), &w, &h, &n, 3);

    if (img == NULL) {
        exit(1);
    }

    vector<vector<float>> data; data.reserve(w * h);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            data.push_back({
                (float) img[(y * w + x) * 3 + 0], 
                (float) img[(y * w + x) * 3 + 1], 
                (float) img[(y * w + x) * 3 + 2]
            });
        }
    }

    stbi_image_free(img);

    return data;
}




std::vector<std::vector<float>> getRandomCentroids(const std::vector<std::vector<float>>& dataset, int k) {
    std::vector<std::vector<float>> newCentroids; newCentroids.reserve(k);

    for (int i = 0; i < k; ++i) {
        newCentroids.push_back(rng::choice(dataset));
    }

    return newCentroids;
}


template <class Method = BasicLloydIteration, typename = std::enable_if_t<std::is_base_of_v<LloydIteration, Method>>>
double timePerIteration(const std::vector<std::vector<float>>& dataset, int k, const std::vector<std::vector<float>>& initialCentroids) {

    KMeans model(k);

    // needed for other stuff besides initializing centroids (I just removed the initialization itself for benchmarks)
    model.initializeCentroids(dataset);

    // set centroids that will actually be used
    model.centroids = initialCentroids;

    Timer timer{};
    timer.start();

    int iter = model.fit<Method>(50); // limit to 50 iterations so it never takes too long

    timer.stop();
    double timePerIteration = timer.elapsedMilliseconds() / ((double) iter);
    
    return timePerIteration;
}








int main() {

    string src;
    cout << "Image src: "; cin >> src;

    std::vector<std::vector<float>> data = getImageData(src);



    /********************************************************************
    *                                                                   *
    *               benchmark on different values of K:                 *
    *                                                                   *
    ********************************************************************/


    std::vector<int> Ks = { 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256 };

    // just to print the same initial centroids and use them in other implementations
/*  rng::setSeed(123);
    for (auto& k : Ks) {
        cout << getRandomCentroids(data, k) << ", ";
    }
*/

    std::vector<double> timesBasic;
    std::vector<double> timesSIMD;
    std::vector<double> timesOMP;
    std::vector<double> timesOMPSIMD;

    rng::setSeed(123);
    for (auto& k : Ks) {
        std::vector<std::vector<float>> initialCentroids = getRandomCentroids(data, k);

        timesBasic.push_back(timePerIteration<BasicLloydIteration>(data, k, initialCentroids));
        timesSIMD.push_back(timePerIteration<SIMDLloydIteration>(data, k, initialCentroids));
        timesOMP.push_back(timePerIteration<ParallelLloydIteration>(data, k, initialCentroids));
        timesOMPSIMD.push_back(timePerIteration<ParallelSIMDLloydIteration>(data, k, initialCentroids));
    }

    cout << "Ks: " << Ks << "\n";
    cout << "timesBasic: " << timesBasic << "\n";
    cout << "timesSIMD: " << timesSIMD << "\n";
    cout << "timesOMP: " << timesOMP << "\n";
    cout << "timesOMPSIMD: " << timesOMPSIMD << "\n";




    /********************************************************************
    *                                                                   *
    *             benchmark on different number of threads              *
    *                                                                   *
    ********************************************************************/


/*  std::vector<int> numThreads = { 1, 2, 3, 4 };
    int K = 16;

    std::vector<double> timesOMP;
    std::vector<double> timesOMPSIMD;

    rng::setSeed(123);
    std::vector<std::vector<float>> initialCentroids = getRandomCentroids(data, K);

//  cout << initialCentroids << "\n";

    for (auto& k : numThreads) {

        omp_set_num_threads(k);

        timesOMP.push_back(timePerIteration<ParallelLloydIteration>(data, K, initialCentroids));
        timesOMPSIMD.push_back(timePerIteration<ParallelSIMDLloydIteration>(data, K, initialCentroids));
    }

    cout << "num threads: " << numThreads << "\n";
    cout << "timesOMP: " << timesOMP << "\n";
    cout << "timesOMPSIMD: " << timesOMPSIMD << "\n";*/


    return 0;
}
