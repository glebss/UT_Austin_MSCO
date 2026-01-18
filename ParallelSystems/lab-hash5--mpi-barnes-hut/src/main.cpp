#include <mpi.h>
#include <iostream>
#include "argparse.h"
#include "force.h"
#include "io.h"
#include "tree.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    options_t opts;
    get_opts(argc, argv, &opts);
    Particle* objects = nullptr;  
    int num_particles = 0;
    if (world_rank == 0) {
        num_particles = read_data(opts.in_file, &objects);
    }
    double start_time, end_time;
    if (world_rank == 0) {
        start_time = MPI_Wtime();
    }
    MPI_Bcast(&num_particles, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (num_particles < world_size) {
        throw std::runtime_error{"Please set th number of processes <= numn_particles"};
    }

    // decide for each process on which part of particles to work
    int num_particles_chunk = num_particles / world_size;
    int start_idx = world_rank * num_particles_chunk;
    int end_idx = (world_rank + 1) * num_particles_chunk;
    if (world_rank == world_size - 1) {
        end_idx = num_particles;
    }
    int local_recv_count = (end_idx - start_idx) * sizeof(Particle);
    int local_displ = start_idx * sizeof(Particle);
    int* recv_counts = new int[world_size];
    int* displs = new int[world_size];
    MPI_Allgather(&local_recv_count, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&local_displ, 1, MPI_INT, displs, 1, MPI_INT, MPI_COMM_WORLD);

    if (world_rank != 0) {
        objects = new Particle[num_particles];
    }
    MPI_Bcast(objects, num_particles * sizeof(Particle), MPI_BYTE, 0, MPI_COMM_WORLD);
    for (int n_step = 0; n_step < opts.n_steps; ++n_step) {
        Tree tree;
        construct_tree(objects, num_particles, &tree);
        calculate_batch(objects, start_idx, end_idx, &tree, opts.theta, opts.dt);
        MPI_Barrier(MPI_COMM_WORLD);
        Particle* send_buffer = new Particle[local_recv_count];
        std::memcpy(send_buffer, objects + start_idx, local_recv_count);
        MPI_Allgatherv(send_buffer, local_recv_count, MPI_BYTE,
                       objects, recv_counts, displs, MPI_BYTE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (world_rank == 0) {
        end_time = MPI_Wtime();
        std::cout << end_time - start_time << std::endl;
    }
    if (world_rank == 0) {
        write_data(opts.out_file, objects, num_particles);
    }

    delete[] objects;
    delete[] displs;
    delete[] recv_counts;

    MPI_Finalize();
    return 0;
}