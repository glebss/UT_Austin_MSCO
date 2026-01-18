#include "kmeans_cuda.cuh"
#include <cuda_runtime.h>

__device__ int converged_device;


__global__ void zero_out_centroids(float* centroids, int num_centroids, int num_features) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x < num_centroids && idx_y < num_features) {
        centroids[idx_x * num_features + idx_y] = 0.f;
    }
}

__global__ void accumulate_centroids(float* data, float* centroids, int* labels,
                                     int num_points, int num_features) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        int label = labels[idx];
        for (int j = 0; j < num_features; ++j) {
            atomicAdd(&centroids[label * num_features + j], data[idx * num_features + j]);
        }
    }
    __syncthreads();
}

__global__ void average_centroids(float* centroids, int* num_labels, int num_centroids, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_centroids) {
        int count = num_labels[idx];
        for (int i = 0; i < num_features; ++i) {
            centroids[idx * num_features + i] /= (float) count;
        }
    }
}

__global__ void zero_out_num_labels(int* num_labels, int num_centroids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_centroids) {
        num_labels[idx] = 0;
    }
}

__global__ void find_min_distance_2d(float* distances, int num_points, int num_centroids, int* min_labels, int* num_labels) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_x < num_points) {
        float min_dist = distances[idx_x * num_centroids];
        int min_ind = 0;
        float cur_dist = 0.f;
        for (int i = 0; i < num_centroids; ++i) {
            cur_dist = distances[idx_x * num_centroids + i];
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                min_ind = i;
            }
        }
        min_labels[idx_x] = min_ind;
        atomicAdd(&num_labels[min_ind], 1);
    }
}

__global__ void calculate_distances_2d(float* data, float* centroids, float* distances,
                                       int num_points, int num_centroids, int num_features) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x < num_points && idx_y < num_centroids) {
        float distance = 0.f;
        float d = 0.f;
        for (int i = 0; i < num_features; ++i) {
            d = data[idx_x * num_features + i] - centroids[idx_y * num_features + i];
            distance += d * d;
        }
        distances[idx_x * num_centroids + idx_y] = sqrt(distance);
    }
}


__global__ void compare_centroids(float* centroids, float* old_centroids,
                                  int num_centroids, int num_features, float* threshold) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    float d = 0.f;
    if (idx_x < num_centroids && idx_y < num_features) {
        d = centroids[idx_x * num_features + idx_y] - old_centroids[idx_x * num_features + idx_y];
        if (abs(d) > *threshold) {
            atomicAnd(&converged_device, 0);
        }
    }
    
}

void find_nearest_centroids_cuda(float* data, float* centroids, int* labels_device, float* distances_device, int* num_labels_device,
                                 int num_points, int num_features, int num_centroids, dim3 grid_size, dim3 block_size) {

    calculate_distances_2d<<<grid_size, block_size>>>(data, centroids, distances_device, num_points, num_centroids, num_features);
    zero_out_num_labels<<<(num_centroids + NUM_THREADS_X - 1) / NUM_THREADS_X, NUM_THREADS_X>>>(num_labels_device, num_centroids);
    find_min_distance_2d<<<(num_points + NUM_THREADS_PTS - 1) / NUM_THREADS_PTS, NUM_THREADS_PTS>>>
                (distances_device, num_points, num_centroids, labels_device, num_labels_device);
}

return_cuda kmeans_cuda(float* data, float* centroids, int* labels, 
                        int num_points, options_t* args) {
    auto num_features = args->num_features;
    auto num_centroids = args->num_centroids;

    int num_iters = 0;
    int converged;

    float* distances_data2centroids = new float[num_points * num_centroids];
    float* distances_data2centroids_device;
    cudaMalloc((void**)&distances_data2centroids_device, num_points * num_centroids * sizeof(float));

    float* data_device;
    cudaMalloc((void**)&data_device, num_points * num_features * sizeof(float));
    cudaMemcpy(data_device, data, num_points * num_features * sizeof(float), cudaMemcpyHostToDevice);

    float* centroids_device;
    cudaMalloc((void**)&centroids_device, num_centroids * num_features * sizeof(float));
    cudaMemcpy(centroids_device, centroids, num_centroids * num_features * sizeof(float), cudaMemcpyHostToDevice);

    float* old_centroids_device;
    cudaMalloc((void**)&old_centroids_device, num_centroids * num_features * sizeof(float));

    int* labels_device;
    cudaMalloc((void**)&labels_device, num_points * sizeof(int));
    cudaMemcpy(labels_device, labels, num_points * sizeof(int), cudaMemcpyHostToDevice);

    int* num_labels_device;
    cudaMalloc((void**)&num_labels_device, num_centroids * sizeof(int));

    dim3 grid_size_centroids((num_centroids + NUM_THREADS_X - 1) / NUM_THREADS_X, (num_features + NUM_THREADS_Y - 1) / NUM_THREADS_Y);
    dim3 block_size_centroids(NUM_THREADS_X, NUM_THREADS_Y);

    dim3 grid_size_points((num_points + NUM_THREADS_X - 1) / NUM_THREADS_X, (num_centroids + NUM_THREADS_Y - 1) / NUM_THREADS_Y);
    dim3 block_size_points(NUM_THREADS_X, NUM_THREADS_Y);

    float* threshold_device;
    cudaMalloc((void**)&threshold_device, sizeof(float));
    cudaMemcpy(threshold_device, &args->threshold, sizeof(float), cudaMemcpyHostToDevice);


    float elapsed_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (true) {
        cudaMemcpy(old_centroids_device, centroids_device, num_centroids * num_features * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        find_nearest_centroids_cuda(data_device, centroids_device, labels_device,
                        distances_data2centroids_device, num_labels_device, num_points, num_features,
                        num_centroids, grid_size_points, block_size_points);
        
        zero_out_centroids<<<grid_size_centroids, block_size_centroids>>>(centroids_device, num_centroids, num_features);
        accumulate_centroids<<<(num_points + NUM_THREADS_PTS - 1) / NUM_THREADS_PTS, NUM_THREADS_PTS>>>
                    (data_device, centroids_device, labels_device, num_points, num_features);
        average_centroids<<<(num_centroids + NUM_THREADS_X - 1) / NUM_THREADS_X, NUM_THREADS_X>>>
                    (centroids_device, num_labels_device, num_centroids, num_features);
        
        converged = 1;
        cudaMemcpyToSymbol(converged_device, &converged, sizeof(int));

        compare_centroids<<<grid_size_centroids, block_size_centroids>>>(centroids_device, old_centroids_device,
                                                   num_centroids, num_features, threshold_device);

        cudaMemcpyFromSymbol(&converged, converged_device, sizeof(int));

        num_iters++;
        if (converged || num_iters >= args->max_num_iter) {
            break;
        }
        
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(data_device);
    cudaFree(old_centroids_device);
    cudaMemcpy(centroids, centroids_device, num_centroids * args->num_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(centroids_device);
    cudaMemcpy(labels, labels_device, num_points * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(labels_device);
    cudaFree(num_labels_device);
    cudaFree(distances_data2centroids_device);
    cudaFree(threshold_device);

    delete[] distances_data2centroids;

    return {num_iters, elapsed_time};
}