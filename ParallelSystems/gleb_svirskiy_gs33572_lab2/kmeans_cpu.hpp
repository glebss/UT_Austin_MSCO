#pragma once

#include "utils.hpp"
#include "argparse.hpp"

class Dataset {
public:
    Dataset() {}

    int get_num_points() {
        return num_points_;
    }

    void set_data(int num_points, int num_features) {
        num_points_ = num_points;
        data_ = (float *)malloc(num_points * num_features * sizeof(float));
    }

    float* get_data() const {
        return data_;
    }

    ~Dataset() {
        free(data_);
    }

private:
    float* data_;
    int num_points_;
};

void find_nearest_centroids(float* data, float* centroids, int* labels,
                            int num_points, int num_features, int num_centroids);

void average_labeled_centroids(float* data, float* centroids, int* labels,
                               int num_points, int num_features, int num_centroids);

bool compare_centroids(float* centroids, float* old_centroids, int num_centroids, int num_features);


int kmeans_cpu(Dataset& dataset, float* centroids, int* labels, options_t* args);