# -*- coding: utf-8 -*-
"""
@author: S. Cai
"""

from __future__ import absolute_import

import warnings
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from torch.nn.init import calculate_gain
from torch.autograd import Function
from torch.autograd import Variable 
from torch.nn.parameter import Parameter
from torch.hub import load_state_dict_from_url

import torch.linalg as linalg

from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.helpers import build_model_with_cfg

import math
import numpy as np



############# for material encoding module #############
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


##########
class VecAxTrans(nn.Module):
    def __init__(self, mod2dTo1d=True):
        super(VecAxTrans, self).__init__()
        self.mod2dTo1d = mod2dTo1d

    def forward(self, x):
        
        if self.mod2dTo1d:
            x = x.squeeze(-1).transpose(-1, -2) # 2-D to 1-D for spatial extend
            
        else:
            x = x.transpose(-1, -2).unsqueeze(-1) # 1-D to 2-D for spatial extend
            
        return x
    
    
    
class ParamGNorm(nn.Module):
    def __init__(self, H=1, W=1, AxTrans=False, param=True, w=1., k=0.):
        super(ParamGNorm, self).__init__()
        if param:
            if AxTrans:
                points = int(H*W)
                self.gamma = Parameter(w*torch.ones(1, points, 1))
                self.beta = Parameter(k*torch.ones(1, points, 1))
            else:
                self.gamma = Parameter(w*torch.ones(1, 1, H, W))
                self.beta = Parameter(k*torch.ones(1, 1, H, W))
        
        self.AxTrans = AxTrans
        self.param = param

    def forward(self, x):  
        if self.param:
            if self.AxTrans:
                x = (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + 1e-12)
                x = self.gamma * x + self.beta
            
            else:
                x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True) + 1e-12)
                x = self.gamma * x + self.beta
        else:
            if self.AxTrans:
                x = (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + 1e-12)
            
            else:
                x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True) + 1e-12)
            
        return x



class LayerGNorm(nn.Module):
    def __init__(self, dim=1, AxTrans=False, param=True, w=1., k=0.):
        super(LayerGNorm, self).__init__()
        if param:
            if AxTrans:
                self.gamma = Parameter(w*torch.ones(1, 1, dim)) # [b,l,c]
                self.beta = Parameter(k*torch.ones(1, 1, dim))
            else:
                self.gamma = Parameter(w*torch.ones(1, dim, 1, 1)) # [b,c,h,w]
                self.beta = Parameter(k*torch.ones(1, dim, 1, 1))
            
        self.AxTrans = AxTrans
        self.param = param

    def forward(self, x):  
        if self.param:
            if self.AxTrans:
                x = (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + 1e-12)
                x = self.gamma * x + self.beta
            
            else:
                x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True) + 1e-12)
                x = self.gamma * x + self.beta
        else:
            if self.AxTrans:
                x = (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + 1e-12)
            
            else:
                x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True) + 1e-12)
            
        return x
    


class LayerGNormMS(nn.Module):
    def __init__(self, dim=1, param=True, w=1., k=0., l=6):
        super(LayerGNormMS, self).__init__()
        if param:
            self.gamma = Parameter(w*torch.ones(1, l, dim, 1, 1)) # [b,l,c,h,w]
            self.beta = Parameter(k*torch.ones(1, l, dim, 1, 1))
            
        self.param = param

    def forward(self, x):  
        if self.param:
            x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True) + 1e-12)
            x = self.gamma * x + self.beta
        else:
            x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True) + 1e-12)
            
        return x



class ChOverlapAvg(nn.Module):
    def __init__(self, kernel_size=32, reduct_rate=16):
        super(ChOverlapAvg, self).__init__()
        self.pad = nn.ReflectionPad1d(padding=(kernel_size//2, 0)) # left padding only
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=reduct_rate)

        self.kernel_size = kernel_size
        self.reduct_rate = reduct_rate

    def forward(self, x):  
        b,c,l = x.shape
        
        if l < self.kernel_size:
            x = F.avg_pool1d(l)(x)
        else:
            if self.kernel_size > self.reduct_rate:
                x = self.avg(self.pad(x))
            else:
                x = self.avg(x)
                
        return x

##############




# This module is adopted from Decoupled Dynamic Filter Networks: https://github.com/theFoxofSky/ddfnet (CVPR 2021)
class FilterNorm(nn.Module):
    def __init__(self, in_channels, kernel_size, filter_type,
                 nonlinearity='linear', running_std=False, running_mean=False):
        assert filter_type in ('spatial', 'channel')
        assert in_channels >= 1
        super(FilterNorm, self).__init__()
        self.in_channels = in_channels
        self.filter_type = filter_type
        self.runing_std = running_std
        self.runing_mean = running_mean
        std = calculate_gain(nonlinearity) / kernel_size
        if running_std:
            self.std = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2) * std, requires_grad=True)
        else:
            self.std = std
        if running_mean:
            self.mean = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2), requires_grad=True)

    def forward(self, x):
        if self.filter_type == 'spatial':
            b, _, h, w = x.size()
            x = x.view(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-12)
            x = x.reshape(b, _, h, w)
            if self.runing_std:
                x = x * self.std[None, :, None, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None, None]
        elif self.filter_type == 'channel':
            b, h, w = x.size(0), x.size(2), x.size(3)
            #l = int(h*w)
            c = self.in_channels
            x = x.view(b, c, -1)
            x = x - x.mean(dim=2, keepdim=True)
            x = x / (x.std(dim=2, keepdim=True) + 1e-12)
            
            if self.runing_std:
                x = x * self.std[None, :, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None]
            x = x.view(b,c,h,w)
        else:
            raise RuntimeError('Unsupported filter type {}'.format(self.filter_type))
        return x
#####################






# SAMixer's definition
class SAMixer(nn.Module):
    def __init__(self, dim=128, spatial=128, rate_reduct=1, branches=4, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU):
        super(SAMixer, self).__init__()
                
        self.spatial = spatial

        self.sign = 'SAMixer' #
        
        # fusion computation
        group_channels = 64
        self.g = dim // group_channels
        self.d = dim // self.g
        
        T = 2 
        self.T = T
        Ws = T #int(T*T)
        self.paths = branches + 1
        self.avg_local1 = nn.Sequential(
                                 nn.Conv2d(dim, dim, kernel_size=Ws, stride=Ws, groups=dim, bias=False),
                                 nn.GroupNorm(num_groups=dim, num_channels=dim, eps=1e-12),
                                 nn.Conv2d(dim, dim, kernel_size=1, bias=True)
                                     ) # sr conv like avg enco
        
        self.avg_local2 = nn.Sequential(
                                 nn.Conv2d(dim, dim, kernel_size=Ws, stride=Ws, groups=dim, bias=False),
                                 nn.GroupNorm(num_groups=dim, num_channels=dim, eps=1e-12),
                                 nn.Conv2d(dim, dim, kernel_size=1, bias=True)
                                     ) # sr conv like avg enco
            
        
        self.norm0 = nn.LayerNorm(dim, eps=1e-12) #LayerGNorm(dim=dim, AxTrans=True)
        self.norm1 = nn.LayerNorm(dim, eps=1e-12) #LayerGNorm(dim=dim, AxTrans=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-12) #LayerGNorm(dim=dim, AxTrans=True)
        self.norm3 = nn.LayerNorm(dim, eps=1e-12) #LayerGNorm(dim=dim, AxTrans=True)
        self.norm4 = nn.LayerNorm(dim, eps=1e-12) #LayerGNorm(dim=dim, AxTrans=True)
        
        
        self.q = nn.Linear(dim, dim, bias=True) 
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        
        self.scl = (self.d)**-0.5
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.proj = nn.Linear(dim, dim, bias=True)
        
        self.pos_deco = nn.Sequential(
                                    FilterNorm(dim, 1, 'channel', 'relu', running_std=True),
                                    nn.Conv2d(dim, int(T*T), kernel_size=1, bias=True)
                                    ) # param list: 'in_channels' (heads), 'kernel_size' (win_size), 'type', 'nonlinearity' 
        
        
        # form-1: 1*1 dense -> 3*3 dw
        self.norm_inner = LayerGNorm(dim=dim) # 2d norm
        
        expand_ratio = 2
        self.mlp = nn.Sequential(
                                 nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
                                 nn.GroupNorm(num_groups=dim, num_channels=dim, eps=1e-12),
                                 nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, bias=True),
                                 act_layer(),
                                 nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, bias=True)
                                     )
        
        self.norm_outer = nn.LayerNorm(dim, eps=1e-12)

        self.proj_top = nn.Sequential(
                                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                                norm_layer(dim, eps=1e-12), 
                                act_layer()
                                         )

        # init
        self.reset_parameters()
    
    
    @torch.jit.script
    def combine_mul(x, attn):
        return x * attn  
    
    @torch.jit.script
    def combine_add(x, attn):
        return x + attn 
        
    @torch.jit.script
    def combine_add_inputs(x1, x2, x3, x4):
        return F.interpolate( F.interpolate(F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=False) \
                                            + x2, scale_factor=2., mode='bilinear', align_corners=False) + x3, scale_factor=2., mode='bilinear', align_corners=False) + x4
    

    def forward(self, x1, x2, x3, x4): 
        b, c, h, w = x1.shape
        
        ## parametric adaptation (shift computation)
        T, g, d = self.T, self.g, c // self.g

        x = self.combine_add_inputs(x1, x2, x3, x4) # main feature, [b, c, h*8, w*8]

        ## avg enco
        # partition windows
        H, W = h*4, w*4
        N = H * W
        B = b*N
        
        # MHSA
        box0 = self.norm0(self.avg_local1(x).view(b, 1, c, N).permute(0, 3, 1, 2).reshape(B, 1, c))
        q = self.q(box0).view(B, 1, g, d).transpose(1, 2) * self.scl # only x serves as the q, [B, g, 1, d]
        kv0 = self.kv(box0).view(B, 1, 2, g, d).permute(2, 0, 3, 1, 4) # [2, B, g, 1, d]

        kv1 = F.interpolate(self.kv(self.norm1(x1.view(b, c, N//16).transpose(1, -1).contiguous())).transpose(1, -1).reshape(b, 2*c, h, w), scale_factor=4)\
            .view(b, 1, 2*c, N).permute(0, 3, 1, 2).reshape(B, 1, 2, g, d).permute(2, 0, 3, 1, 4)

        kv2 = F.interpolate(self.kv(self.norm2(x2.view(b, c, N//4).transpose(1, -1).contiguous())).transpose(1, -1).reshape(b, 2*c, h*2, w*2), scale_factor=2)\
            .view(b, 1, 2*c, N).permute(0, 3, 1, 2).reshape(B, 1, 2, g, d).permute(2, 0, 3, 1, 4)

        kv3 = self.kv(self.norm3(x3.view(b, 1, c, N).permute(0, 3, 1, 2).reshape(B, 1, c))).view(B, 1, 2, g, d).permute(2, 0, 3, 1, 4) # [2, B, g, 1, d]

        kv4 = self.kv(self.norm4(self.avg_local2(x4).view(b, 1, c, N).permute(0, 3, 1, 2).reshape(B, 1, c))).view(B, 1, 2, g, d).permute(2, 0, 3, 1, 4) # [2, B, g, 1, d]

        
        kv = torch.cat([kv0, kv1, kv2, kv3, kv4], dim=3) # [2, B, g, l, d]
        k, v = kv[0], kv[1] # [B, g, l, d]
        
        
        sim = q @ k.transpose(-2, -1) # b g 1 d, b g l c -> b g 1 l 
        sim = self.softmax(sim)
        u_attn = (sim @ v).transpose(1, 2).reshape(B, 1, c) # b g i j, b g j c -> b g i c (i=1, j=4) 
        
        # position decoding
        u_attn = self.proj(u_attn).view(b, H, W, c).permute(0, 3, 1, 2) # [b, c, h, w]
        
        
        u_attn = self.combine_add(u_attn.unsqueeze(2), self.pos_deco(u_attn).unsqueeze(1) )\
            .view(b, c, T, T, H, W).permute(0, 1, 4, 2, 5, 3).reshape(b, c, H*T, W*T) # [b, c, 1, h, w], [b, 1, T*T, h, w], [b, c, T*T, h, w]
        

        # FFN
        x = self.combine_add(x, u_attn)
        x = self.proj_top(self.combine_add(x, self.norm_outer(self.mlp(self.norm_inner(x)).flatten(2).transpose(1,-1) ).transpose(1,-1).reshape(b, c, H*T, W*T) ) )

        return x

    ############################
    def reset_parameters(self):        
        # conv and bn init
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')     
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
#################








class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

        # init
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
        print('decoder init')
        
        
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            print('loaded')

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x




class SAMixerHead(nn.Module):
    """
    """
    def __init__(self, in_channels=[64, 128, 320, 512], feature_strides=[4, 8, 16, 32], 
                 embedding_dim=768, norm_layer=nn.BatchNorm2d, num_classes=20, 
                 in_index=[0, 1, 2, 3], dropout_ratio=0.1, input_transform='multiple_select', align_corners=False):
        super(SAMixerHead, self).__init__()
        self.feature_strides = feature_strides
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels


        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.mixer = SAMixer(dim=embedding_dim, rate_reduct=1, branches=4, norm_layer=norm_layer)
        
        self.dropout = nn.Dropout2d(dropout_ratio) 
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        
        # init
        self.apply(self._init_weights)
        
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
        print('decoder init')
        
        
        
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            print('loaded')
                


    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs        



    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.mixer(_c4, _c3, _c2, _c1) 

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
    
#################



########################## ops functions ##########################

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)