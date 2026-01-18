#include <spin_barrier.h>
#include <thread>

spin_barrier::spin_barrier(int n_threads) : n_threads_{n_threads} {}

int spin_barrier::fetch_and_increment() {
    return counter.fetch_add(1);
}

void spin_barrier::wait() {
    int counter_local = fetch_and_increment();
    bool go_local = go.load();
    if (counter_local + 1 == n_threads_) {
        counter.store(0);
        go.store(!go.load());
    } else {
        while(go_local == go.load()) {}
    }
    return;
}
