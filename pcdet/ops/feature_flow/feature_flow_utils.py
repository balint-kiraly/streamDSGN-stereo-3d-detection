import torch
import torch.autograd
import torch.nn.functional as F

from pcdet.ops.feature_flow import feature_flow_cuda


def offset_matching_gpu(sweeping_feature, adjusting_feature, shift_range):
    """
    Args:
        sweeping_feature: (B, C, H, W)
        adjusting_feature: (B, C, H, W)
        shift_range: (D, 2)
        matching_threshold: float

    Returns:
        matching_dist: (B, D, C, H, W)
    """

    assert sweeping_feature.size() == adjusting_feature.size()
    assert shift_range.size(1) == 2
    
    B, C, H, W = sweeping_feature.size()
    D = shift_range.size(0)
    # Cosine similarity and CUDA atomic operations do not support half precision.
    use_amp = True if sweeping_feature.dtype == torch.float16 else False  #
    matching_dist = []
    for i in range(sweeping_feature.size(0)):
        if use_amp:
            sweeping_batch = sweeping_feature[i, ...].float()
            adjusting_batch = adjusting_feature[i, ...].float()
        else:
            sweeping_batch = sweeping_feature[i, ...]
            adjusting_batch = adjusting_feature[i, ...]

        adjusting_batch_expand = adjusting_batch.unsqueeze(
            0).expand(D, -1, -1, -1)

        sweeping_batch_expand = feature_flow_cuda.feature_sweeping_gpu(
            sweeping_batch, shift_range)
        cosine_sim = F.cosine_similarity(
            adjusting_batch_expand, sweeping_batch_expand, dim=1, eps=1e-6)
        matching_dist_batch = torch.softmax(cosine_sim, dim=0)
        matching_dist.append(matching_dist_batch)
    matching_dist = torch.stack(matching_dist, dim=0)
    return matching_dist


def get_shift_coord(matching_dist, shift_range, matching_threshold=None):
    def argmax_with_threshold(x, dim=0, threshold=0.03, default_index=24):
        max_indices = torch.argmax(x, dim=dim)
        max_values = torch.max(x, dim=dim).values
        condition = max_values > threshold
        result_indices = torch.where(condition, max_indices, default_index)
        return result_indices

    if matching_threshold is None:
        matching_threshold = 1 / (len(shift_range) * 0.8)
        # matching_threshold = 1 / len(shift_range)

    H, W = matching_dist.size(2), matching_dist.size(3)
    shift_index = []
    for i in range(matching_dist.size(0)):
        matching_dist_bacth = matching_dist[i, ...]
        shift_index_batch = argmax_with_threshold(
            matching_dist_bacth, dim=0, threshold=matching_threshold, default_index=int(len(shift_range) / 2))
        shift_index.append(shift_index_batch)
    shift_index = torch.stack(shift_index, dim=0)
    shift_coord = torch.gather(shift_range, 0, shift_index.view(-1,1).expand(-1,2))
    shift_coord = shift_coord.view(1, H, W, 2).permute(0, 3, 1, 2).contiguous()
    shift_coord[:, [0, 1], :, :] = shift_coord[:, [1, 0], :, :]
    return shift_coord


class FeatureAdjustingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, shift_coord, feature_padding=False, with_grad=True):
        """
        Args:
            feature: (B, C, H, W)
            shift_coord: (2, H, W)

        Returns:
            adjusted_feature: (B, C, H, W)
        """
        # B, C, H, W = feature.size()
        # D = shift_range.size(0)
        use_amp = True if feature.dtype == torch.float16 else False
        feature_padding = torch.tensor(feature_padding)
        with_grad = torch.tensor(with_grad)
    
        adjusted_feature = []
        shift_counter_list = []
        for i in range(feature.size(0)):
            if use_amp:
                feature_batch = feature[i, ...].float()
            else:
                feature_batch = feature[i, ...]
      
            shift_counter = torch.zeros_like(shift_coord[i, 0, ...]).int()
            adjusted_feature_batch = feature_flow_cuda.feature_adjusting_forward(
                feature_batch, shift_coord[i, ...], shift_counter)
            if feature_padding:
                adjusted_feature_batch[:, shift_counter ==
                                    0] = feature_batch[:, shift_counter == 0]
            if use_amp:
                adjusted_feature_batch = adjusted_feature_batch.half()
            shift_valid = shift_counter.clone()
            shift_counter[shift_counter == 0] = 1
            adjusted_feature_batch /= shift_counter
            adjusted_feature.append(adjusted_feature_batch)
            shift_counter_list.append(shift_counter)
        adjusted_feature = torch.stack(adjusted_feature, dim=0)
        shift_counter = torch.stack(shift_counter_list, dim=0)
        ctx.save_for_backward(shift_coord, shift_counter, shift_valid, feature_padding, with_grad)
        return adjusted_feature

    @staticmethod
    def backward(ctx, grad_output):
        shift_coord, shift_counter, shift_valid, feature_padding, with_grad = ctx.saved_tensors
        if not with_grad:
            return None, None, None, None
        use_amp = True if grad_output.dtype == torch.float16 else False
        grad_feature = []
        for i in range(grad_output.size(0)):
            if use_amp:
                grad_output_batch = grad_output[i, ...].float()
            else:
                grad_output_batch = grad_output[i, ...]

            grad_feature_batch = feature_flow_cuda.feature_adjusting_backward(
                grad_output_batch, shift_coord[i, ...], shift_counter)
            if feature_padding:
                grad_feature_batch[:, shift_valid ==
                                    0] = grad_output_batch[:, shift_valid == 0]

            if use_amp:
                grad_feature_batch = grad_feature_batch.half()
            grad_feature.append(grad_feature_batch)
        grad_feature = torch.stack(grad_feature, dim=0)
        return grad_feature, None, None, None


feature_adjusting = FeatureAdjustingFunction.apply

if __name__ == '__main__':
    import torch.nn as nn
    from pcdet.ops.feature_flow.backward_test import get_example_data, get_shift_range

    last_data, curr_data, next_data = get_example_data(
        size=(1, 10, 12, 12), num_feature=3)
    last_data = last_data.cuda().half()
    curr_data = curr_data.cuda().half()
    next_data = next_data.cuda().half()

    import numpy as np
    last_data = torch.from_numpy(
        np.load('./features/0001_7_000306.npy')).cuda().half()
    last_gt = np.load('./features/000120_gt.npy').squeeze()
    curr_data = torch.from_numpy(
        np.load('./features/0001_7_000307.npy')).cuda().half()
    curr_gt = np.load('./features/000121_gt.npy').squeeze()
    next_data = torch.from_numpy(
        np.load('./features/0001_7_000308.npy')).cuda().half()
    next_gt = np.load('./features/000122_gt.npy').squeeze()
    last_data_pool = F.max_pool2d(last_data, kernel_size=2, stride=2)
    curr_data_pool = F.max_pool2d(curr_data, kernel_size=2, stride=2)
    next_data_pool = F.max_pool2d(next_data, kernel_size=2, stride=2)
    # last_data = last_data[:, :24, ...]
    # curr_data = curr_data[:, :24, ...]
    # next_data = next_data[:, :24, ...]

    matching_range = (-3, 3, -3, 3)
    shift_range = get_shift_range(matching_range).cuda()
    matching_dist = offset_matching_gpu(
        last_data_pool, curr_data_pool, shift_range)
    shift_coord = get_shift_coord(matching_dist, shift_range).float()

    up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
    shift_coord_bi = up_sample(shift_coord) * 2

    up_sample = nn.Upsample(scale_factor=2, mode='nearest')
    shift_coord_ne = up_sample(shift_coord) * 2

    np.save('last_data.npy', last_data.cpu().numpy())
    np.save('curr_data.npy', curr_data.cpu().numpy())
    np.save('next_data.npy', next_data.cpu().numpy())
    np.save('shift_coord_bi.npy', shift_coord_bi.cpu().numpy())
    np.save('shift_coord_ne.npy', shift_coord_ne.cpu().numpy())
