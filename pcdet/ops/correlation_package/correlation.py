import torch
from torch.autograd import Function
from torch.nn.modules.module import Module

from pcdet.ops.correlation_package import correlation_cuda


class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx, fmap1, fmap2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply):
        assert fmap1.is_contiguous() and fmap2.is_contiguous()
        ctx.save_for_backward(fmap1, fmap2)

        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        with torch.cuda.device_of(fmap1):
            rbot1 = fmap1.new()
            rbot2 = fmap2.new()
            output = fmap1.new()
            correlation_cuda.forward(fmap1, fmap2, rbot1, rbot2, output, pad_size,
                                     kernel_size, max_displacement, stride1, stride2, corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        fmap1, fmap2 = ctx.saved_tensors

        pad_size = ctx.pad_size
        kernel_size = ctx.kernel_size
        max_displacement = ctx.max_displacement
        stride1 = ctx.stride1
        stride2 = ctx.stride2
        corr_multiply = ctx.corr_multiply

        with torch.cuda.device_of(fmap1):
            rbot1 = fmap1.new()
            rbot2 = fmap2.new()

            grad_input1 = fmap1.new()
            grad_input2 = fmap2.new()

            correlation_cuda.backward(fmap1, fmap2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                                      pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)

        return grad_input1, grad_input2, None, None, None, None, None, None


correlation_function = CorrelationFunction.apply


class Correlation(Module):
    def __init__(self, pad_size=3, kernel_size=1, max_displacement=3, stride1=1, stride2=1, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):

        result = correlation_function(input1, input2, self.pad_size, self.kernel_size,
                                      self.max_displacement, self.stride1, self.stride2, self.corr_multiply)

        return result


if __name__ == '__main__':
    import time
    corr_module = Correlation(
        pad_size=5, kernel_size=1, max_displacement=5, stride1=1, stride2=1, corr_multiply=1)

    fmap1 = torch.randn(1, 96, 288, 256).cuda()
    fmap2 = torch.randn(1, 96, 288, 256).cuda()
    for i in range(50):
        torch.cuda.synchronize()
        t1 = time.time()
        output = corr_module(fmap1, fmap2)
        torch.cuda.synchronize()
        t2 = time.time()
        print(t2 - t1)
    print(output.shape)
