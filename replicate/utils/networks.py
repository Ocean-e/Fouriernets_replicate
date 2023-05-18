import math
import torch
import torch.cuda.comm
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from itertools import repeat
from .unet.model import FUNet2D


def FourierNetRGB(
    fourier_out,
    fourier_kernel_size,
    fourier_conv_args=None,
    conv_kernel_sizes=[11],
    conv_fmap_nums=None,
    input_scaling_mode="batchnorm",
    quantile=0.5,
    quantile_scale=1.0,
    imbn_momentum=0.1,
    fourierbn_momentum=0.1,
    convbn_momentum=0.1,
    device="cpu",
):
    # ensure kernel sizes and fmap nums are lists
    conv_kernel_sizes = _list(conv_kernel_sizes) #转化成list
    conv_kernel_sizes = [_pair(k) for k in conv_kernel_sizes] #如果size已经有（a,b)，则直接输出；若为[a,b,c]形式，则输出[(a,a),(b,b),(c,c)]
    assert conv_fmap_nums[-1] == 3, "Must output 3 features (RGB)" #错误时触发 "Must output 3 features (RGB)"
    # create image batch norm
    if input_scaling_mode is None or input_scaling_mode in ["scaling", "norming"]:
        layers = []
    elif input_scaling_mode in ["batchnorm"]:
        imbn = nn.BatchNorm2d(3, momentum=imbn_momentum).to(device)
        layers = [("image_bn", imbn)]
    else:
        raise TypeError(
            "Argument input_scaling_mode must be one of None, 'scaling', 'norming', or 'batchnorm'"
        )
    if fourier_conv_args is None:
        fourier_conv_args = {}
    # create fourier convolution and batch norm
    fourier_conv = FourierConv2D(
        3, fourier_out, fourier_kernel_size, **fourier_conv_args
    ).to(device)
    fourier_relu = nn.LeakyReLU().to(device)
    fourierbn = nn.BatchNorm2d(fourier_out, momentum=fourierbn_momentum).to(device)
    layers += [
        ("fourier_conv", fourier_conv),
        ("fourier_relu", fourier_relu),
        ("fourier_bn", fourierbn),
    ]
    # create convolution layers
    # if multiple convolution layers, use LeakyReLU until last layer
    # use batch norms between convolutions (after activation)
    # ReLU on final layer for non-negative output, no batch norm at end
    for i in range(len(conv_fmap_nums)): #conv_fmap_nums确定卷积每一层的输入输出通道数
        previous_fmap_nums = (
            fourier_out if i == 0 else conv_fmap_nums[i - 1]
        )
        conv = nn.Conv2d(
            previous_fmap_nums, #in_channels
            conv_fmap_nums[i], #out_channels
            conv_kernel_sizes[i], #kernel_size
            padding=[int(math.floor(k / 2)) for k in conv_kernel_sizes[i]], #每个维度pad kernelsize的一半 #padding=[5,5]
        )
        conv = conv.to(device)
        layers.append((f"conv2d_{i+1}", conv))
        if i < len(conv_fmap_nums) - 1:
            conv_relu = nn.LeakyReLU().to(device)
            layers.append((f"conv{i+1}_relu", conv_relu))
            conv_bn = nn.BatchNorm2d(conv_fmap_nums[i], momentum=convbn_momentum).to(
                device
            )
            layers.append((f"conv{i+1}_bn", conv_bn))
        else:
            conv_relu = nn.ReLU().to(device)
            layers.append((f"conv{i+1}_relu", conv_relu))

    # construct and return sequential module [找不到InputScalingSequential函数，跳过]
    
    #if input_scaling_mode == "scaling":
    #    reconstruct = InputScalingSequential(
    #        quantile, quantile_scale, OrderedDict(layers)
    #    )
    #elif input_scaling_mode == "norming":
    #    reconstruct = InputNormingSequential((0, 2, 3), OrderedDict(layers))
    #else:

    reconstruct = nn.Sequential(OrderedDict(layers))
    return reconstruct


class FourierConv2D(nn.Module):
    """
    Applies a 2D Fourier convolution over an input signal composed of several
    input planes. That is, 2D convolution is performed by performing a Fourier
    transform, multiplying a kernel and the input, then taking the inverse
    Fourier transform of the result.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=True,
        bias=True,
        reduce_channels=True,
        real_feats_mode="index",
    ):
        super(FourierConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = padding
        # kernel size depends on whether we pad or not
        if self.padding and self.stride == (1, 1):
            self.kernel_size = (kernel_size[0] * 2 - 1, kernel_size[1] * 2 - 1)
        else:
            self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size, 2)
        )
        self.reduce_channels = reduce_channels
        if bias and self.reduce_channels:
            self.bias = nn.Parameter(torch.Tensor(out_channels, 1, 1))
        elif bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels, 1, 1, 1))
        else:
            self.register_parameter("bias", None)
        self.real_feats_mode = real_feats_mode
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    def forward(self, im):
        # determine input, output, and fourier domain sizes
        insize = im.size()
        outsize = (self.out_channels, im.size(-2), im.size(-1))
        fftsize = [insize[-2] + outsize[-2] - 1, insize[-1] + outsize[-1] - 1]

        # construct zero pad for real signal if needed
        if self.padding:
            pad_size = (
                0,
                1,
                0,
                fftsize[-1] - insize[-1],
                0,
                fftsize[-2] - insize[-2],
            )
        else:
            pad_size = (0, 1)
        # add channel for imaginary component and perform FFT
        fourier_im = torch.fft(F.pad(im.unsqueeze(-1), pad_size), 2)

        if self.stride != (1, 1):
            # skip inputs if stride > 1
            stride = self.stride
            fourier_im = fourier_im[:, :, :: stride[-2], :: stride[-1], :]

        # calculate features in fourier space
        # (add dimension for batch broadcasting)
        fourier_feats = cmul2(fourier_im.unsqueeze(1), self.weight)

        # retrieve real component of signal
        if self.real_feats_mode == "index":
            indices = torch.tensor(0, device=fourier_im.device)
            real_feats = torch.ifft(fourier_feats, 2).index_select(-1, indices)
            real_feats = real_feats.squeeze(-1)
        elif self.real_feats_mode == "abs":
            real_feats = cabs(torch.ifft(fourier_feats, 2))
        else:
            return NotImplemented

        # crop feature maps back to original size if we padded
        if self.padding and self.stride == (1, 1):
            cropsize = [fftsize[-2] - insize[-2], fftsize[1] - insize[-1]]
            cropsize_left = [int(c / 2) for c in cropsize]
            cropsize_right = [int((c + 1) / 2) for c in cropsize]
            real_feats = F.pad(
                real_feats,
                (
                    -cropsize_left[-1],
                    -cropsize_right[-1],
                    -cropsize_left[-2],
                    -cropsize_right[-2],
                ),
            )

        # sum over input channels to get correct number of output channels
        if self.reduce_channels:
            real_feats = real_feats.sum(2)

        # add bias term
        real_feats = real_feats + self.bias

        return real_feats
    
def FourierUNetRGB(
    in_channels,
    out_channels,
    in_shape,
    scale_factors=[1, 2, 4, 8],
    funet_fmaps=[64, 64, 64, 64],
    conv_kernel_size=(11, 11, 11),
    conv_padding=(1, 2, 2),
    funet_kwargs={},
    input_scaling_mode="batchnorm",
    imbn_momentum=0.5,
    quantile=0.5,
    quantile_scale=1.0,
    device="cpu",
):
    # create image batch norm
    if input_scaling_mode in ["scaling", "norming"]:
        layers = []
    elif input_scaling_mode in ["batchnorm"]:
        imbn = nn.BatchNorm2d(1, momentum=imbn_momentum).to(device)
        layers = [("image_bn", imbn)]
    else:
        raise TypeError(
            "Argument input_scaling_mode must be one of 'scaling', 'norming', or 'batchnorm'"
        )
    # create funet
    funet = FUNet2D(
        in_channels,
        out_channels,
        in_shape,
        scale_factors,
        f_maps=funet_fmaps,
        conv_kernel_size=conv_kernel_size,
        conv_padding=conv_padding,
        **funet_kwargs,
    ).to(device)
    funet_relu = nn.ReLU().to(device)
    layers += [("funet", funet), ("funet_relu", funet_relu)]
    # construct and return sequential module
    #if input_scaling_mode == "scaling":
    #    reconstruct = InputScalingSequential(
    #        quantile, quantile_scale, OrderedDict(layers)
    #    )
    #elif input_scaling_mode == "norming":
    #    reconstruct = InputNormingSequential((0, 2, 3), OrderedDict(layers))
    #else:
    reconstruct = nn.Sequential(OrderedDict(layers))
    return reconstruct


# functions for calculation

def cmul2(x, y, xy=None):
    xr, xi = real(x), imag(x)
    yr, yi = real(y), imag(y)
    xyr = torch.mul(xr, yr) - torch.mul(xi, yi)
    xyi = torch.mul(xr, yi) + torch.mul(xi, yr)
    xy = torch.stack([xyr, xyi], -1, out=xy)
    return xy

def real(x):
    return x.index_select(-1, torch.tensor(0, device=x.device)).squeeze(-1)

def imag(x):
    return x.index_select(-1, torch.tensor(1, device=x.device)).squeeze(-1)

def cabs(x):
    xabs = x.pow(2).sum(-1).sqrt()
    return xabs

def _list(x, repetitions=1):
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return x
    else:
        return [
            x,
        ] * repetitions

def _ntuple(n):
    """Creates a function enforcing ``x`` to be a tuple of ``n`` elements."""

    def parse(x): 
        if isinstance(x, tuple):
            return x
        return tuple(repeat(x, n))

    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)