#pragma once

struct options_t {
    char* in_file;
    char* out_file;
    int n_steps;
    double theta;
    double dt;
    bool visualize{false};
};

void get_opts(int argc, char** argv, options_t* opts);