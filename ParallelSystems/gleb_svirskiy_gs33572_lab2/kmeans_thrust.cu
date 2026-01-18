#include "kmeans_thrust.cuh"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <float.h>

__device__ int converged_device;

struct find_nearest_centroid_functor_t
{
    thrust::device_ptr<float> data_;
    thrust::device_ptr<float> centroids_;
    int num_centroids_;
    int num_features_;
    thrust::device_ptr<int> labels_;
    thrust::device_ptr<int> num_labels_;
    
    explicit find_nearest_centroid_functor_t(thrust::device_ptr<float> data,
                                             thrust::device_ptr<float> centroids,
                                             int num_centroids, int num_features,
                                             thrust::device_ptr<int> labels,
                                             thrust::device_ptr<int> num_labels) {
        data_ = data;
        centroids_ = centroids;
        num_centroids_ = num_centroids;
        num_features_ = num_features;
        labels_ = labels;
        num_labels_ = num_labels;
    }

    __device__
    void operator()(int data_idx) {
        float min_dist = FLT_MAX;
        int min_centroid = -1;
        for (int i = 0; i < num_centroids_; ++i) {
            float cur_dist = 0.f;
            float d = 0.f;
            for (int j = 0; j < num_features_; ++j) {
                d = data_[data_idx * num_features_ + j] - centroids_[i * num_features_ + j];
                cur_dist += d * d;
            }
            cur_dist = sqrt(cur_dist);
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                min_centroid = i;
            }
        }
        labels_[data_idx] = min_centroid;
        atomicAdd(thrust::raw_pointer_cast(&num_labels_[min_centroid]), 1);
    }
};

struct zero_out_centroids_functor_t
{
    thrust::device_ptr<float> centroids_;
    int num_features_;

    explicit zero_out_centroids_functor_t(thrust::device_ptr<float> centroids,
                                          int num_features) {
        centroids_ = centroids;
        num_features_ = num_features;
    }

    __device__
    void operator()(int centroid_idx) {
        for (int i = 0; i < num_features_; ++i) {
            centroids_[centroid_idx * num_features_ + i] = 0.f;
        }
    }
};

struct zero_out_num_labels_functor_t
{
    thrust::device_ptr<int> num_labels_;

    explicit zero_out_num_labels_functor_t(thrust::device_ptr<int> num_labels) {
        num_labels_ = num_labels;
    }

    __device__
    void operator()(int centroid_idx) {
        num_labels_[centroid_idx] = 0;
    }
};

struct accumulate_centroids_functor_t
{
    thrust::device_ptr<float> data_;
    thrust::device_ptr<float> centroids_;
    int num_features_;
    thrust::device_ptr<int> labels_;
    
    explicit accumulate_centroids_functor_t(thrust::device_ptr<float> data,
                                           thrust::device_ptr<float> centroids,
                                           int num_features,
                                           thrust::device_ptr<int> labels) {
        data_ = data;
        centroids_ = centroids;
        num_features_ = num_features;
        labels_ = labels;
    }
    
    __device__
    void operator()(int data_idx) {
        int label = labels_[data_idx];
        for (int j = 0; j < num_features_; ++j) {
            atomicAdd(thrust::raw_pointer_cast(&centroids_[label * num_features_ + j]),
                      data_[data_idx * num_features_ + j]);
        }
    }
};

struct average_centroids_functor_t
{
    thrust::device_ptr<float> centroids_;
    thrust::device_ptr<int> num_labels_;
    int num_features_;

    explicit average_centroids_functor_t(thrust::device_ptr<float> centroids,
                                         thrust::device_ptr<int> num_labels,
                                         int num_features) {
        centroids_ = centroids;
        num_labels_ = num_labels;
        num_features_ = num_features;
    }

    __device__
    void operator()(int centroid_idx) {
        float count = (float) num_labels_[centroid_idx];
        for (int i = 0; i < num_features_; ++i) {
            centroids_[centroid_idx * num_features_ + i] = centroids_[centroid_idx * num_features_ + i] / count;
        }
    }
};

struct compare_centroids_functor_t
{
    
    thrust::device_ptr<float> centroids_;
    thrust::device_ptr<float> old_centroids_;
    int num_features_;
    float threshold_;
    
    explicit compare_centroids_functor_t(thrust::device_ptr<float> centroids,
                                         thrust::device_ptr<float> old_centroids,
                                         int num_features,
                                         float threshold) {
        centroids_ = centroids;
        old_centroids_ = old_centroids;
        num_features_ = num_features;
        threshold_ = threshold;
    }
    
    __device__
    void operator()(int num_centroid) {
        float d = 0.f;
        for (int j = 0; j < num_features_; ++j) {
            d = centroids_[num_centroid * num_features_ + j] - old_centroids_[num_centroid * num_features_ + j];
            if (abs(d) > threshold_) {
                atomicAnd(&converged_device, 0);
            }
        }
    }
};


return_thrust kmeans_thrust(float* data, float* centroids, int* labels, 
                            int num_points, options_t* args) {

    auto num_features = args->num_features;
    auto num_centroids = args->num_centroids;

    int num_iters = 0;
    int converged;

    float* data_device;
    cudaMalloc((void**)&data_device, num_points * num_features * sizeof(float));
    cudaMemcpy(data_device, data, num_points * num_features * sizeof(float), cudaMemcpyHostToDevice);
    thrust::device_ptr<float> thrust_data_ptr(data_device);

    float* centroids_device;
    cudaMalloc((void**)&centroids_device, num_centroids * num_features * sizeof(float));
    cudaMemcpy(centroids_device, centroids, num_centroids * num_features * sizeof(float), cudaMemcpyHostToDevice);
    thrust::device_ptr<float> thrust_centroids_ptr(centroids_device);

    float* old_centroids_device;
    cudaMalloc((void**)&old_centroids_device, num_centroids * num_features * sizeof(float));
    thrust::device_ptr<float> thrust_old_centroids_ptr(old_centroids_device);

    int* labels_device;
    cudaMalloc((void**)&labels_device, num_points * sizeof(int));
    thrust::device_ptr<int> thrust_labels_ptr(labels_device);

    int* num_labels_device;
    cudaMalloc((void**)&num_labels_device, num_centroids * sizeof(int));
    thrust::device_ptr<int> thrust_num_labels_ptr(num_labels_device);

    thrust::counting_iterator<int> data_idx_begin(0);
    thrust::counting_iterator<int> data_idx_end(num_points);

    thrust::counting_iterator<int> centroids_idx_begin(0);
    thrust::counting_iterator<int> centroids_idx_end(num_centroids);

    zero_out_num_labels_functor_t zero_out_num_labels_functor(
                    thrust_num_labels_ptr);

    find_nearest_centroid_functor_t find_nearest_centroid_functor(
                    thrust_data_ptr, thrust_centroids_ptr,
                    num_centroids, num_features, thrust_labels_ptr,
                    thrust_num_labels_ptr);
    
    zero_out_centroids_functor_t zero_out_centroids_functor(
                    thrust_centroids_ptr, num_features);
    
    accumulate_centroids_functor_t accumulate_centroids_functor(
                    thrust_data_ptr, thrust_centroids_ptr,
                    num_features, thrust_labels_ptr);
    
    average_centroids_functor_t average_centroids_functor(
                    thrust_centroids_ptr, thrust_num_labels_ptr,
                    num_features);
    
    compare_centroids_functor_t compare_centroids_functor(
                    thrust_centroids_ptr, thrust_old_centroids_ptr,
                    num_features, args->threshold);

    float elapsed_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (true) {
        thrust::copy(thrust::device, thrust_centroids_ptr,
                     thrust_centroids_ptr + num_centroids * num_features, thrust_old_centroids_ptr);

        thrust::for_each(thrust::device, centroids_idx_begin, centroids_idx_end, zero_out_num_labels_functor);
        
        thrust::for_each(thrust::device, data_idx_begin, data_idx_end, find_nearest_centroid_functor);

        thrust::for_each(thrust::device, centroids_idx_begin, centroids_idx_end, zero_out_centroids_functor);

        thrust::for_each(thrust::device, data_idx_begin, data_idx_end, accumulate_centroids_functor);

        thrust::for_each(thrust::device, centroids_idx_begin, centroids_idx_end, average_centroids_functor);

        converged = 1;
        cudaMemcpyToSymbol(converged_device, &converged, sizeof(int));

        thrust::for_each(thrust::device, centroids_idx_begin, centroids_idx_end, compare_centroids_functor);

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

    cudaMemcpy(labels, thrust::raw_pointer_cast(thrust_labels_ptr), num_points * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(labels_device);

    cudaMemcpy(centroids, thrust::raw_pointer_cast(thrust_centroids_ptr), num_centroids * num_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(centroids_device);

    cudaFree(data_device);
    cudaFree(old_centroids_device);
    cudaFree(num_labels_device);

    return {num_iters, elapsed_time};
}