import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import correlation_cuda  # 确保这个 CUDA 扩展可用


class CorrelationFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply):
        # ✅ 使用 ctx 代替 self 进行状态存储
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        with torch.cuda.device_of(input1):
            rbot1 = input1.new_empty(0)
            rbot2 = input2.new_empty(0)
            output = input1.new_empty(0)

            correlation_cuda.forward(
                input1, input2, rbot1, rbot2, output,
                pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply
            )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # ✅ 从 ctx 取出保存的状态
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new_empty(0)
            rbot2 = input2.new_empty(0)

            grad_input1 = input1.new_empty(0)
            grad_input2 = input2.new_empty(0)

            correlation_cuda.backward(
                input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply
            )

        return grad_input1, grad_input2, None, None, None, None, None, None


class Correlation(nn.Module):
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        # ✅ 现在正确地使用 `apply()` 来调用静态 `autograd.Function`
        return CorrelationFunction.apply(
            input1, input2, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply
        )
