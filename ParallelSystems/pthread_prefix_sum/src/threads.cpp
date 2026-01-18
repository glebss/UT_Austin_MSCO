#include "threads.h"
#include "helpers.h"

pthread_t *alloc_threads(int n_threads)
{
  return (pthread_t *)malloc(n_threads * sizeof(pthread_t));
}

void start_threads(pthread_t *threads,
                   int n_threads,
                   struct prefix_sum_args_t *args,
                   void *(*start_routine)(void *)) {
  int ret = 0;
  for (int i = 0; i < n_threads; ++i) {
    ret |= pthread_create(&(threads[i]), NULL, start_routine,
                          (void *)&(args[i]));
  }
  if (ret) {
    std::cerr << "Error starting threads" << std::endl;
    exit(1);
  }
}

void join_threads(pthread_t *threads,
                  int n_threads) {
  int res = 0;
  for (int i = 0; i < n_threads; ++i) {
    int *ret;
    res |= pthread_join(threads[i], (void **)&ret);
  }

  if (res) {
    std::cerr << "Error joining threads" << std::endl;
    exit(1);
  }
}

void* compute_prefix_sum_thread(void* args) {
    prefix_sum_args_t* args_t = (prefix_sum_args_t*) args;
    int start = args_t->t_id * args_t->num_elems_per_thread;
    int end = (args_t->t_id + 1) * args_t->num_elems_per_thread;
    bool spin = args_t->spin;
    
    // First sweep
    args_t->output_vals[start] = args_t->input_vals[start];
    for (int i = start + 1; i < end; ++i) {
        args_t->output_vals[i] = args_t->op(args_t->output_vals[i-1], args_t->input_vals[i], args_t->n_loops);
    }

    // Barrier
    int ret;
    if (spin) {
      args_t->s_barrier->wait();
    } else {
      ret = pthread_barrier_wait(args_t->barrier);
      if (ret != 0 && ret != PTHREAD_BARRIER_SERIAL_THREAD) {
          std::cerr << "Error waiting for barrier" << std::endl;
          exit(1);
      }
    }
    
    // Total sum accumulation
    if (args_t->t_id == 0) {
      int x = 0;
      int n = 0;
      for (int i = 0; i < args_t->n_threads; ++i) {
        n = (i + 1) * args_t->num_elems_per_thread - 1;
        x += args_t->output_vals[n];
        args_t->output_vals[n+1] = x;
      }
    }

    // Barrier
    if (spin) {
      args_t->s_barrier->wait();
    } else {
      ret = pthread_barrier_wait(args_t->barrier);
      if (ret != 0 && ret != PTHREAD_BARRIER_SERIAL_THREAD) {
          std::cerr << "Error waiting for barrier" << std::endl;
          exit(1);
      }
    }

    // Second sweep
    start += args_t->num_elems_per_thread;
    end += args_t->num_elems_per_thread;
    if (args_t->t_id == args_t->n_threads - 1) {
      end = args_t->n_vals;
    }
    args_t->output_vals[start] += args_t->input_vals[start];
    for (int i = start + 1; i < end; ++i) {
      args_t->output_vals[i] = args_t->op(args_t->output_vals[i-1], args_t->input_vals[i], args_t->n_loops);
    }

    return 0;
}
