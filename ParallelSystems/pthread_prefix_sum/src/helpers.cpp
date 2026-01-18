#include "helpers.h"
#include <math.h>
#include <iostream>

prefix_sum_args_t* alloc_args(int n_threads) {
  return (prefix_sum_args_t*) malloc(n_threads * sizeof(prefix_sum_args_t));
}

int next_power_of_two(int x) {
    int pow = 1;
    while (pow < x) {
        pow *= 2;
    }
    return pow;
}

void fill_args(prefix_sum_args_t *args,
               int n_threads,
               int n_vals,
               int *inputs,
               int *outputs,
               bool spin,
               int (*op)(int, int, int),
               int n_loops,
               pthread_barrier_t* barrier,
               spin_barrier* s_barrier) {
    if (spin && !s_barrier) {
        std::cerr << "Spin barrier is nullptr. Please check your arguments" << std::endl;
        exit(1);
    }
    for (int i = 0; i < n_threads; ++i) {
        if (!spin) {
            args[i] = {inputs, outputs, spin, n_vals,
                   n_threads, i, op, n_loops, barrier, NULL,
                  (int) floor((double)n_vals / (n_threads + 1))};
        } else {
            args[i] = {inputs, outputs, spin, n_vals,
                   n_threads, i, op, n_loops, NULL, s_barrier,
                  (int) floor((double)n_vals / (n_threads + 1))};
        }
        
    }
}