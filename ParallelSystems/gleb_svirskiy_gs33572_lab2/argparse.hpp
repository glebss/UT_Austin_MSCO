#pragma once

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {
    int num_centroids;
    int num_features;
    char* in_file;
    int max_num_iter;
    float threshold;
    bool out_centroids;
    int seed;
    bool use_cuda;
    bool use_thrust;
};

void get_opts(int argc, char **argv, struct options_t *opts);
