#include "utils.hpp"


int kmeans_rand() {
    NEXT = NEXT * 1103515245 + 12345;
    return (unsigned int)(NEXT / 65536) % (KMEANS_RMAX+1);
}

void kmeans_srand(unsigned int seed) {
    NEXT = seed;
}

void get_random_centroids(float* data, float* centroids,
                          int num_centroids, int num_features, int num_points) {
    for (int i = 0; i < num_centroids; ++i) {
        int index = kmeans_rand() % num_points;
        for (int j = 0; j < num_features; ++j) {
            centroids[i * num_features + j] = data[index * num_features + j];
        }
    }
}

float calc_dist(float* data_vec, float* centroid, int num_features) {
    float dist = 0.f;
    for (int i = 0; i < num_features; ++i) {
        auto d = centroid[i] - data_vec[i];
        dist += d * d;
    }
    return sqrt(dist);
}