#pragma once

#include "argparse.hpp"

struct return_thrust {
    int num_iters;
    float elapsed_time;
};

return_thrust kmeans_thrust(float* data, float* centroids, int* labels,
                            int num_points, options_t* args);