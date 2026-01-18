#ifndef _SPIN_BARRIER_H
#define _SPIN_BARRIER_H

#include <pthread.h>
#include <atomic>

class spin_barrier {
public:    
    spin_barrier(int n_threads);
    void wait();

private:
    int fetch_and_increment();
    std::atomic<int> counter{0};
    std::atomic<bool> go{true};
    int n_threads_;
};

#endif
