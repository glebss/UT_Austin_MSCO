#pragma once

#include <cmath>

static unsigned long int NEXT = 1;
static unsigned long KMEANS_RMAX = 32767;

int kmeans_rand();

void kmeans_srand(unsigned int seed);

void get_random_centroids(float* dataset, float* centroids,
                          int num_centroids, int num_features, int num_points);

float calc_dist(float* data_vec, float* centroid, int num_features);