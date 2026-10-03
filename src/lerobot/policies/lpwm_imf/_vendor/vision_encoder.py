# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2023 Tal Daniel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Extracted from modules/vision_modules.py at LPWM 4cf53c403433e64c01652ac2adbec66231a46dea.
# ruff: noqa
# fmt: off

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    # lrelu
    # return F.leaky_relu(x, negative_slope=0.01)
    # relu
    # return F.relu(x)
    # gelu
    return F.gelu(x)


def norm_layer(in_channels, num_groups=4, eps=1e-5):
    # base_groups = num_groups
    # if in_channels <= 32:
    #     num_groups = base_groups
    # elif in_channels == 64:
    #     num_groups = base_groups * 2  # 8
    # elif num_groups == 128:
    #     num_groups = base_groups * 4  # 16
    # else:
    #     num_groups = base_groups * 8  # 32
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, use_conv_block=False, padding_mode='constant'):
        super().__init__()
        self.with_conv = with_conv
        self.use_conv_block = use_conv_block
        self.padding_mode = 'constant' if padding_mode == 'zeros' else padding_mode
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            if self.use_conv_block:
                self.conv = ConvBlock(in_channels=in_channels, out_channels=in_channels, dropout=0.0, padding=0,
                                      stride=2, kernel_size=3)
            else:
                self.conv = torch.nn.Conv2d(in_channels,
                                            in_channels,
                                            kernel_size=3,
                                            stride=2,
                                            padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode=self.padding_mode, value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ConvBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, temb_channels=0, padding_mode='zeros', padding=1, stride=1, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding, padding_mode=padding_mode)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm = norm_layer(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, temb=None):
        h = x
        h = self.conv(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        return h


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=0, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = norm_layer(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1, padding_mode=padding_mode)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = norm_layer(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1, padding_mode=padding_mode)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1, padding_mode=padding_mode)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = norm_layer(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(self, *, ch, ch_mult=(1, 2, 4, 8), num_res_blocks, residual=True,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, padding_mode='zeros', attention=False,
                 mid_blocks=True, in_conv_kernel_size=3, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.padding_mode = padding_mode
        self.use_attention = attention
        self.residual = residual
        self.mid_blocks = mid_blocks
        block_nn = ResnetBlock if self.residual else ConvBlock

        # downsampling
        # if self.residual:
        #     self.conv_in = torch.nn.Conv2d(in_channels,
        #                                    self.ch,
        #                                    kernel_size=3,
        #                                    stride=1,
        #                                    padding=1, padding_mode=self.padding_mode)
        # else:
        #     self.conv_in = ConvBlock(in_channels=in_channels, out_channels=self.ch, padding_mode=self.padding_mode,
        #                              temb_channels=self.temb_ch, dropout=dropout)

        first_conv_pad = in_conv_kernel_size // 2
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=in_conv_kernel_size,
                                       stride=1,
                                       padding=first_conv_pad, padding_mode=self.padding_mode)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(block_nn(in_channels=block_in,
                                      out_channels=block_out,
                                      temb_channels=self.temb_ch,
                                      dropout=dropout, padding_mode=self.padding_mode))
                block_in = block_out
                if curr_res in attn_resolutions:
                    if attention:
                        attn.append(AttnBlock(block_in))
                    else:
                        attn.append(nn.Identity())
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                # down.downsample = Downsample(block_in, resamp_with_conv, padding_mode=padding_mode,
                #                              use_conv_block=not self.residual)
                down.downsample = Downsample(block_in, resamp_with_conv, padding_mode=padding_mode)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()

        if self.mid_blocks:
            self.mid.block_1 = block_nn(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout, padding_mode=self.padding_mode)
            if attention:
                self.mid.attn_1 = AttnBlock(block_in)
            else:
                self.mid.attn_1 = nn.Identity()
            self.mid.block_2 = block_nn(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout, padding_mode=self.padding_mode)
        else:
            self.mid.block_1 = nn.Identity()
            self.mid.attn_1 = nn.Identity()
            self.mid.block_2 = nn.Identity()

        # if attention:
        #     self.mid.block_1 = block_nn(in_channels=block_in,
        #                                 out_channels=block_in,
        #                                 temb_channels=self.temb_ch,
        #                                 dropout=dropout, padding_mode=self.padding_mode)
        #     self.mid.attn_1 = AttnBlock(block_in)
        #     self.mid.block_2 = block_nn(in_channels=block_in,
        #                                 out_channels=block_in,
        #                                 temb_channels=self.temb_ch,
        #                                 dropout=dropout, padding_mode=self.padding_mode)
        # else:
        #     self.mid.block_1 = nn.Identity()
        #     self.mid.attn_1 = nn.Identity()
        #     self.mid.block_2 = nn.Identity()

        # end
        self.norm_out = norm_layer(block_in) if self.residual else nn.Identity()
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1, padding_mode=self.padding_mode)
        self.conv_output_size = self.calc_conv_output_size()

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.in_channels, self.resolution, self.resolution)
        dummy_input = self(dummy_input)
        return dummy_input[0].shape

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        if self.mid_blocks:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)

        # end
        if self.residual:
            h = self.norm_out(h)
            h = nonlinearity(h)
        h = self.conv_out(h)
        return h
