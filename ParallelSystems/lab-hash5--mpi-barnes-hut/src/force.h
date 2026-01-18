#pragma once
#include "tree.h"

struct Force {
    double Fx;
    double Fy;
};

void calculate_total_force_and_update_particle(Particle* particle, Tree* tree, double theta, double dt,
                                               double G=1e-4, double rlimit=3e-2);

void calculate_batch(Particle* objects, int start_idx, int end_idx, Tree* tree, double theta, double dt);