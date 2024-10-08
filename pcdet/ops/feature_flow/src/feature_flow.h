#ifndef FEATURE_FLOW_H
#define FEATURE_FLOW_H

#include <torch/extension.h>


at::Tensor feature_adjusting_forward(
    at::Tensor& features,
    at::Tensor& shift_coord,
    at::Tensor& shift_counter
);


at::Tensor feature_adjusting_backward(
    at::Tensor& d_output,
    at::Tensor& shift_coord,
    at::Tensor& shift_counter
);


at::Tensor feature_sweeping_gpu(
    at::Tensor& features,
    at::Tensor& shift_range
);

#endif