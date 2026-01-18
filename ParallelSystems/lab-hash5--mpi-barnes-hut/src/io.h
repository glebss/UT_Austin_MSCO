#pragma once
#include "tree.h"

int read_data(const char* in_file, Particle** objects);

void write_data(const char* out_file, Particle* objects, int num_particles);