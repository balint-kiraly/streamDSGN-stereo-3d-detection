#include <stdint.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/extension.h>


template <typename T>
__global__ void feature_sweeping_cuda_kernel(
    const T* features,      // (C, H, W): (96, 144, 128)
    const int64_t *shift_range, // (D, 2): (49, 2)
    T* output,              // (D, C, H, W): (49, 96, 144, 128)
    int depths,                 // D
    int channels,               // C
    int height,                 // H
    int width)                  // W
{
    int height_idx = blockIdx.x;
    int depth_idx = blockIdx.y;
    int thread_idx = threadIdx.x;

    int shift_h = shift_range[depth_idx * 2];
    int shift_w = shift_range[depth_idx * 2 + 1];

    int new_h = height_idx + shift_h;
    int new_w = thread_idx + shift_w;

    if (new_h < 0 || new_h >= height || new_w < 0 || new_w >= width)
    {
        return;
    }

    for (int c = 0; c < channels; c++)
    {
        int output_index = depth_idx * channels * height * width + c * height * width + new_h * width + new_w;
        int feature_index = c * height * width + height_idx * width + thread_idx;
        output[output_index] = features[feature_index];
    }

    // int height_idx = blockIdx.x;
    // int thread_idx = threadIdx.x;

    // for (int d = 0; d < depths; d++)
    // {
    //     int shift_h = shift_range[d * 2];
    //     int shift_w = shift_range[d * 2 + 1];

    //     int new_h = height_idx + shift_h;
    //     int new_w = thread_idx + shift_w;

    //     if (new_h < 0 || new_h >= height || new_w < 0 || new_w >= width)
    //     {
    //         continue;
    //     }
    //     for (int c = 0; c < channels; c++)
    //     {
    //         int output_index = d * channels * height * width + c * height * width + new_h * width + new_w;
    //         int feature_index = c * height * width + height_idx * width + thread_idx;
    //         output[output_index] = features[feature_index];
    //     }
    // }
}


at::Tensor feature_sweeping_launcher(
    const at::Tensor& features,
    const at::Tensor& shift_range,
    int depths,
    int channels,
    int height,
    int width)
{
    dim3 blocks = {static_cast<unsigned int>(height), static_cast<unsigned int>(depths)};
    dim3 threads = {static_cast<unsigned int>(width)};

    auto output = at::zeros({depths, channels, height, width}, features.options());

    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "feature_sweeping_cuda_kernel", ([&] {
        feature_sweeping_cuda_kernel<scalar_t><<<blocks, threads>>>(
            features.data_ptr<scalar_t>(),
            shift_range.data_ptr<int64_t>(),
            output.data_ptr<scalar_t>(),
            depths,
            channels,
            height,
            width
        );
    }));

    return output;
}