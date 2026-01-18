#include "kmeans_cpu.hpp"


void find_nearest_centroids(float* data, float* centroids, int* labels,
                            int num_points, int num_features, int num_centroids) {
    for (int i = 0; i < num_points; ++i) {
        auto min_dist = calc_dist(data + i * num_features, centroids, num_features);
        int min_label = 0;
        for (int k = 1; k < num_centroids; ++k) {
            auto cur_dist = calc_dist(data + i * num_features, centroids + k * num_features, num_features);
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                min_label = k;
            }
        }
        labels[i] = min_label;
    }
}

void average_labeled_centroids(float* data, float* centroids, int* labels,
                               int num_points, int num_features, int num_centroids) {
    for (int i = 0; i < num_centroids; ++i) {
        for (int j = 0; j < num_features; ++j) {
            centroids[i * num_features + j] = 0.f;
        }
    }

    float* labels_num = new float[num_centroids];
    for (int i = 0; i < num_centroids; ++i) {
        labels_num[i] = 0.f;
    }

    for (int i = 0; i < num_points; ++i) {
        int label = labels[i];
        labels_num[label] += 1.0;
        for (int j = 0; j < num_features; ++j) {
            centroids[label * num_features + j] += data[i * num_features + j];
        }
    }

    for (int i = 0; i < num_centroids; ++i) {
        float num_samples_per_label = labels_num[i];
        for (int j = 0; j < num_features; ++j) {
            centroids[i * num_features + j] /= num_samples_per_label;
        }
    }
    delete[] labels_num;
}

bool compare_centroids(float* centroids, float* old_centroids, int num_centroids, int num_features, float threshold) {
    for (int i = 0; i < num_centroids; ++i) {
        for (int j = 0; j < num_features; ++j) {
            if (abs(centroids[i * num_features + j] - old_centroids[i * num_features + j]) > threshold) {
                return false;
            }
        }
    }
    return true;
}

int kmeans_cpu(Dataset& dataset, float* centroids, int* labels, options_t* args) {
    auto num_features = args->num_features;
    auto num_points = dataset.get_num_points();
    auto num_centroids = args->num_centroids;

    float* old_centroids = new float[num_centroids * num_features];

    int num_iters = 0;
    bool done = false;
    while (!done) {
        for (int i = 0; i < num_centroids; ++i) {
            for (int j = 0; j < num_features; ++j) {
                old_centroids[i * num_features + j] = centroids[i * num_features + j];
            }
        }

        find_nearest_centroids(dataset.get_data(), centroids, labels, num_points, num_features, num_centroids);
        average_labeled_centroids(dataset.get_data(), centroids, labels, num_points, num_features, num_centroids);
        num_iters++;
        done = num_iters >= args->max_num_iter || compare_centroids(centroids, old_centroids, num_centroids, num_features, args->threshold);
    }

    delete[] old_centroids;

    return num_iters;
}