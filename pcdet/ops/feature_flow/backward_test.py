import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
import torch.optim as optim
from torch.autograd import gradcheck

from feature_flow_utils import offset_matching_gpu, get_shift_coord, feature_adjusting


def get_example_data(num_feature=4, offset_range=(-3, 3), size=(1, 32, 20, 20), keep_offset=True):
        last_data, curr_data, next_data = torch.zeros(
            size=size), torch.zeros(size=size), torch.zeros(size=size)
        B, C, H, W = size
        torch.manual_seed(520)
        coord = torch.randint(
            offset_range[1]+1, min(H, W)+offset_range[0], size=(2, num_feature))
        # torch.manual_seed(0)
        insert_feature = torch.randn(size=(B, C, num_feature))

        if keep_offset:
            offset = torch.randint(
                offset_range[0], offset_range[1]+1, size=(2, num_feature))
            last_coord = coord + offset
            next_coord = coord - offset
        else:
            offset_last = torch.randint(
                offset_range[0], offset_range[1]+1, size=(2, num_feature))
            last_coord = coord + offset_last
            offset_next = torch.randint(
                offset_range[0], offset_range[1]+1, size=(2, num_feature))
            next_coord = coord + offset_next

        for i in range(num_feature):
            curr_data[:, :, coord[0, i], coord[1, i]] = insert_feature[..., i]
            last_data[:, :, last_coord[0, i],
                    last_coord[1, i]] = insert_feature[..., i]
            next_data[:, :, next_coord[0, i],
                    next_coord[1, i]] = insert_feature[..., i]
        print(f'curr_coord:\n {coord}')
        print(f'last_coord:\n {last_coord}')
        print(f'next_coord:\n {next_coord}')
        return last_data, curr_data, next_data


def get_shift_range(matching_range, reverse=False):
        left, right, top, bottom = matching_range
        if reverse:
            shift_range = torch.tensor([(i, j) for i in range(right, left-1, -1)
                                        for j in range(bottom, top-1, -1)])
        else:
            shift_range = torch.tensor([(i, j) for i in range(left, right+1, 1)
                                        for j in range(top, bottom+1, 1)])
        return shift_range


def forward_function(input_data, shift_index, shift_range, next_data):
    adjusted_feature = feature_adjusting(input_data, shift_index, shift_range)
    loss = torch.mean((adjusted_feature - next_data) ** 2)
    return loss, adjusted_feature


def test_shift_index():
    # last_data, curr_data, next_data = get_example_data(
        # size=(1, 10, 20, 20), num_feature=10, offset_range=(-6, 6), keep_offset=True)
    
    last_data = torch.from_numpy(
        np.load('./features/000120.npy')).cuda().half()
    last_gt = np.load('./features/000120_gt.npy').squeeze()
    curr_data = torch.from_numpy(
        np.load('./features/000121.npy')).cuda().half()
    curr_gt = np.load('./features/000121_gt.npy').squeeze()
    next_data = torch.from_numpy(
        np.load('./features/000122.npy')).cuda().half()
    next_gt = np.load('./features/000122_gt.npy').squeeze()
    
    last_data = last_data.cuda().half()
    curr_data = curr_data.cuda().half()
    next_data = next_data.cuda().half()
    
    _last_data = last_data.cpu().numpy()
    _curr_data = curr_data.cpu().numpy()
    _next_data = next_data.cpu().numpy()
    np.save('last_data0.npy', _last_data)
    np.save('curr_data0.npy', _curr_data)
    np.save('next_data0.npy', _next_data)

    last_data1 = F.avg_pool2d(last_data, kernel_size=2, stride=2)
    curr_data1 = F.avg_pool2d(curr_data, kernel_size=2, stride=2)
    next_data1 = F.avg_pool2d(next_data, kernel_size=2, stride=2)

    np.save('last_data1.npy', last_data1.cpu().numpy())
    np.save('curr_data1.npy', curr_data1.cpu().numpy())
    np.save('next_data1.npy', next_data1.cpu().numpy())

    shift_range = (-3, 3, -3, 3)
    shift_range = get_shift_range(shift_range)
    shift_range = shift_range.cuda()
    
    matching_dist = offset_matching_gpu(last_data1, curr_data1, shift_range)
    shift_coord = get_shift_coord(matching_dist, shift_range).float()
    print(shift_coord)
    shift_coord = shift_coord * 2
    shift_coord = F.interpolate(shift_coord, scale_factor=2, mode='nearest').long()
    print(shift_coord)
    adjusted_feature = feature_adjusting(curr_data, shift_coord)
    np.save('adjusted_feature2.npy', adjusted_feature.cpu().numpy())


def test():
    last_data = torch.from_numpy(
        np.load('./features/000120.npy')).cuda().half()
    last_gt = np.load('./features/000120_gt.npy').squeeze()
    curr_data = torch.from_numpy(
        np.load('./features/000121.npy')).cuda().half().requires_grad_()
    curr_gt = np.load('./features/000121_gt.npy').squeeze()
    next_data = torch.from_numpy(
        np.load('./features/000122.npy')).cuda().half()
    next_gt = np.load('./features/000122_gt.npy').squeeze()
    shift_range = (-3, 3, -3, 3)
    shift_range = get_shift_range(shift_range)
    shift_range = shift_range.cuda()

    last_data1 = F.avg_pool2d(last_data, kernel_size=2, stride=2)
    curr_data1 = F.avg_pool2d(curr_data, kernel_size=2, stride=2)
    next_data1 = F.avg_pool2d(next_data, kernel_size=2, stride=2)
    matching_dist = offset_matching_gpu(last_data1, curr_data1, shift_range)
    shift_coord = get_shift_coord(matching_dist, shift_range).float() * 2
    shift_coord = F.interpolate(shift_coord, scale_factor=2, mode='nearest').long()
    adjusted_feature = feature_adjusting(curr_data, shift_coord, True)
    criterion = nn.MSELoss()
    loss = criterion(adjusted_feature, next_data)
    print(loss)
    loss.backward()



if __name__ == '__main__':
    #  main()
    test()
    # test_shift_index()
    
