#include <stdint.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/Dispatch.h>
// #define THREADS_PER_BLOCK 256
// #define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

template <typename T>
__global__ void feature_adjusting_cuda_kernel_forward(
    const T* features,  // (C, H, W): (96, 144, 128)
    const int64_t*  shift_coord,  // (H, W): (2, 144, 128)
    int* shift_counter,  // (H, W): (144, 128)
    T* output,  // (C, H, W): (96, 144, 128)
    int height,
    int width,
    int channels
) 
{
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    // int element_idx = block_idx * width + thread_idx;

    int offset_h = shift_coord[block_idx * width + thread_idx];
    int offset_w = shift_coord[(block_idx + height) * width + thread_idx];
    
    // if (offset_h != 0 && offset_w != 0)
    // {
    //     printf("offset_h: %d, offset_w: %d \n", offset_h, offset_w);
    //     printf("ori_h: %d, ori_w: %d \n", block_idx, thread_idx);
    // }
    
    int new_h = block_idx + offset_h;
    int new_w = thread_idx + offset_w;

    if (new_h < 0 || new_h >= height || new_w < 0 || new_w >= width) {
        return;
    }

    // if (offset_h == 0 && offset_w == 0) {
    //     for (int c=0; c < channels; c++) {
    //         atomicAdd(&output[c * height * width + new_h * width + new_w], features[c * height * width + block_idx * width + thread_idx]);
    //     }
    //     return;
    // }

    atomicAdd(&shift_counter[(new_h * width + new_w)], 1);
    for (int c=0; c < channels; c++) {
        // float* output_address = reinterpret_cast<float*>(&output[c * height * width + new_h * width + new_w]);
        // float feature_value = static_cast<float>(features[c * height * width + block_idx * width + thread_idx]);
        // atomicAdd(output_address, feature_value);
        atomicAdd(&output[c * height * width + new_h * width + new_w], features[c * height * width + block_idx * width + thread_idx]);
    }
    return;
}


template <typename T>
__global__ void feature_adjusting_cuda_kernel_backward(
    const T*  d_adjusted_feature,  // (C, H, W): (96, 144, 128)
    const int64_t*  shift_coord,  // (H, W): (2, 144, 128)
    const int*  shift_counter,  // (H, W): (144, 128)
    T* d_feature,  // (C, H, W): (96, 144, 128)
    int height,
    int width,
    int channels
)
{
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    // int element_idx = block_idx * width + thread_idx;

    int offset_h = shift_coord[block_idx * width + thread_idx];
    int offset_w = shift_coord[(block_idx + height) * width + thread_idx];
    int new_h = block_idx + offset_h;
    int new_w = thread_idx + offset_w;

    if (new_h < 0 || new_h >= height || new_w < 0 || new_w >= width) {
        return;
    }

    int count = shift_counter[new_h * width + new_w];
    if (count == 0) {
        count = 1;
    }
    for (int c=0; c < channels; c++) {
        // float* address = reinterpret_cast<float*>(&d_feature[c * height * width + block_idx * width + thread_idx]);
        // float value = static_cast<float>(d_adjusted_feature[c * height * width + new_h * width + new_w]) / count;
        // atomicAdd(address, value);
        atomicAdd(&d_feature[c * height * width + block_idx * width + thread_idx], d_adjusted_feature[c * height * width + new_h * width + new_w] / count);
    }
}


at::Tensor feature_adjusting_forward_launcher(
    const at::Tensor& features,
    const at::Tensor& shift_coord,
    at::Tensor& shift_counter,
    int height,
    int width,
    int channels
)
{
    // dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));
    // dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks = height;
    dim3 threads = width;
    
    auto output = at::zeros({channels, height, width}, features.options());

    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "feature_adjusting_cuda_kernel_forward", ([&] {
        feature_adjusting_cuda_kernel_forward<scalar_t><<<blocks, threads>>>(
            features.data<scalar_t>(),
            shift_coord.data<int64_t>(),
            shift_counter.data<int>(),
            output.data<scalar_t>(),
            height,
            width,
            channels
        );
    }));

    return output;
}


at::Tensor feature_adjusting_backward_launcher(
    const at::Tensor& d_output,
    const at::Tensor& shift_coord,
    const at::Tensor& shift_counter,
    int height,
    int width,
    int channels
)
{
    dim3 blocks = height;
    dim3 threads = width;

    auto d_feature = at::zeros({channels, height, width}, d_output.options());

    AT_DISPATCH_FLOATING_TYPES(d_output.scalar_type(), "feature_adjusting_cuda_kernel_backward", ([&] {
        feature_adjusting_cuda_kernel_backward<scalar_t><<<blocks, threads>>>(
            d_output.data_ptr<scalar_t>(),
            shift_coord.data_ptr<int64_t>(),
            shift_counter.data_ptr<int>(),
            d_feature.data_ptr<scalar_t>(),
            height,
            width,
            channels
        );
    }));

    return d_feature;
}