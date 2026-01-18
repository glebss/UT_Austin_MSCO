#pragma once

#include "argparse.hpp"

static constexpr int NUM_THREADS_X = 8;
static constexpr int NUM_THREADS_Y = 8;
static constexpr int NUM_THREADS_PTS = 32;

struct return_cuda {
    int num_iters;
    float elapsed_time;
};

return_cuda kmeans_cuda(float* data, float* centroids, int* labels,
                        int num_points, options_t* args);