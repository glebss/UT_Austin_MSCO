#include "argparse.hpp"
#include "utils.hpp"
#include "kmeans_cpu.hpp"
#include "kmeans_cuda.cuh"
#include "kmeans_thrust.cuh"
#include "io.hpp"

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

int main(int argc, char **argv) {
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);
    if (opts.use_cuda && opts.use_thrust) {
        std::cerr << "Please specify either cuda or thrust implementation. Cannot use both at the same time.\n";
        exit(1);
    }

    Dataset dataset;

    read_data(&opts, dataset);

    kmeans_srand(opts.seed);

    float* centroids = new float[opts.num_centroids * opts.num_features];
    int* labels = new int[dataset.get_num_points()];

    get_random_centroids(dataset.get_data(), centroids, opts.num_centroids, opts.num_features, dataset.get_num_points());

    if (!opts.use_cuda && !opts.use_thrust) {
        auto start = std::chrono::high_resolution_clock::now();
        int num_iters = kmeans_cpu(dataset, centroids, labels, &opts);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << num_iters << "," << duration.count() / 1000.f << std::endl;
    } else if (opts.use_cuda) {
        auto ret = kmeans_cuda(dataset.get_data(), centroids, labels,
                               dataset.get_num_points(), &opts);
        std::cout << ret.num_iters << "," << ret.elapsed_time << std::endl;
    } else {
        auto ret = kmeans_thrust(dataset.get_data(), centroids, labels,
                                dataset.get_num_points(), &opts);
        std::cout << ret.num_iters << "," << ret.elapsed_time << std::endl;
    }

    if (opts.out_centroids) {
        for (int i = 0; i < opts.num_centroids; ++i) {
            std::cout << i << " ";
            for (int j = 0; j < opts.num_features; ++j) {
                std::cout << centroids[i * opts.num_features + j] << " ";
            }
            std::cout << '\n';
        }
    } else {
        std::cout << "clusters:";
        for (int i = 0; i < dataset.get_num_points(); ++i) {
            std::cout << " " << labels[i];
        }
        std::cout << '\n';
    }

    delete[] centroids;
    delete[] labels;
    return 0;
}
