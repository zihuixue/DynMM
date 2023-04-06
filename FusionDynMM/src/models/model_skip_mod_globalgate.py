# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet import ResNet18, ResNet34, ResNet50
from src.models.rgb_depth_fusion import SqueezeAndExciteFusionAdd, SqueezeAndExciteReweigh
from src.models.context_modules import get_context_module
from src.models.resnet import BasicBlock, NonBottleneck1D
from src.models.model_utils import ConvBNAct, Swish, Hswish
from src.models.model import Decoder


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class SkipGateESANet(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=40,
                 encoder_rgb='resnet34',
                 encoder_depth='resnet34',
                 encoder_block='NonBottleneck1D',
                 channels_decoder=[128, 128, 128],
                 pretrained_on_imagenet=False,
                 pretrained_dir='./trained_models/imagenet',
                 activation='relu',
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=[3, 3, 3],  # default: [1, 1, 1]
                 fuse_depth_in_rgb_encoder='add',
                 upsampling='learned-3x3-zeropad',
                 temp=1,
                 block_rule=None):  # 0: rgb only, 1: rgb+d, others: dynamic


        super(SkipGateESANet, self).__init__()

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder
        self.block_rule = block_rule if block_rule else [1, 1, 1, 1]  # [0, 0, 0, 0] rgb;  [1, 1, 0, 0] skip last 2 fusion components

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        if encoder_rgb == 'resnet50' or encoder_depth == 'resnet50':
            warnings.warn('Parameter encoder_block is ignored for ResNet50. '
                          'ResNet50 always uses Bottleneck')

        # rgb encoder
        if encoder_rgb == 'resnet18':
            self.encoder_rgb = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet34':
            self.encoder_rgb = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet50':
            self.encoder_rgb = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_rgb. Got {}'.format(encoder_rgb))

        # depth encoder
        if encoder_depth == 'resnet18':
            self.encoder_depth = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet34':
            self.encoder_depth = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet50':
            self.encoder_depth = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                input_channels=1)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_depth. Got {}'.format(encoder_rgb))

        self.channels_decoder_in = self.encoder_rgb.down_32_channels_out

        if fuse_depth_in_rgb_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExciteFusionAdd(
                64, activation=self.activation)
            self.se_layer1 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_4_channels_out,
                activation=self.activation)
            self.se_layer2 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_8_channels_out,
                activation=self.activation)
            self.se_layer3 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_16_channels_out,
                activation=self.activation)
            self.se_layer4 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_32_channels_out,
                activation=self.activation)

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder_rgb.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder_rgb.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder_rgb.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder_rgb.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder_rgb.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder_rgb.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        elif encoder_decoder_fusion == 'None':
            self.skip_layer0 = nn.Identity()
            self.skip_layer1 = nn.Identity()
            self.skip_layer2 = nn.Identity()
            self.skip_layer3 = nn.Identity()

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = \
            get_context_module(
                context_module,
                self.channels_decoder_in,
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

        # gating network
        self.temp = temp
        self.gate_layer = GlobalGate(branch_num=5)
        self.baseline = False
        self.ini_stage = False
        self.hard_gate = False
        self.save_weight_info = False
        self.weight_list = torch.Tensor()
        if encoder_rgb == 'resnet34':
            self.flop = torch.Tensor([0, 3.27, 7.27, 13.15, 16.02]).cuda()
            self.depth_enc_flop = torch.Tensor([0.2506752, 3.1113216, 6.9470208, 12.66432, 15.538944]).cuda()
            self.total_flop = torch.Tensor([22.37101509, 25.23166149, 29.06736069, 34.78465989, 37.65928389]).cuda()
        else:
            self.depth_enc_flop = torch.Tensor([0.2506752, 4.39420573, 10.72382115, 19.71582947, 24.679084]).cuda()
            self.total_flop = torch.Tensor([32.5854654, 36.728995928, 43.058611352, 52.050619672, 57.0138742]).cuda()

    def freeze(self):
        for name, param in self.named_parameters():
            if 'gate' not in name:
                param.requires_grad = False

    def start_weight(self):
        self.save_weight_info = True
        self.weight_list = torch.Tensor()

    def end_weight(self, print_each=False, print_flop=False):
        self.save_weight_info = False
        if print_each:
            print(self.weight_list)
        weight_mean = torch.mean(self.weight_list, axis=0)
        cnt_list = np.zeros(5, dtype=float)
        for i in range(5):
            tmp = self.weight_list[:, i] == 1
            tmp_cnt = tmp.sum()
            tmp_idx = torch.where(tmp)
            # print(f'path {i}, avg val {weight_mean[i]:.2f}, choose cnt {tmp_cnt}')
            # print(tmp_idx)
            if print_flop:
                cnt_list[i] = tmp_cnt
        if print_flop:
            cnt_list = torch.from_numpy(cnt_list / cnt_list.sum()).cuda()
            flop1 = (self.depth_enc_flop * cnt_list).sum()
            flop2 = (self.total_flop * cnt_list).sum()
            print(f'Depth Encoder Flop {flop1:.4f}G | Total Flop {flop2:.4f}G')
        self.weight_list = torch.Tensor()

    def forward(self, rgb, depth, test=False, return_weight=False):
        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(depth)

        fuse = rgb + depth if self.fuse_depth_in_rgb_encoder == 'add' else self.se_layer0(rgb, depth)
        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        bs = rgb.shape[0]
        if self.baseline:
            weight = torch.zeros(bs, 5).cuda()
            weight[:, 4] = 1
        elif self.ini_stage:
            weight = torch.zeros(bs, 5).cuda()
            idx = torch.randint(0, 5, (bs,))
            weight[range(bs), idx] = 1
        else:
            weight = self.gate_layer(rgb, depth, self.temp, self.hard_gate)
        if self.save_weight_info:
            self.weight_list = torch.cat((self.weight_list, weight.cpu()))

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)
        branch0 = rgb
        branch1 = rgb + depth if self.fuse_depth_in_rgb_encoder == 'add' else self.se_layer1(rgb, depth)
        # print(weight.shape)
        w = weight[:, 0].view(-1, 1, 1, 1)
        fuse = w * branch0 + (1 - w) * branch1
        skip1 = self.skip_layer1(fuse)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        branch0 = rgb
        branch1 = rgb + depth if self.fuse_depth_in_rgb_encoder == 'add' else self.se_layer2(rgb, depth)
        w = (weight[:, 0] + weight[:, 1]).view(-1, 1, 1, 1)
        fuse = w * branch0 + (1 - w) * branch1
        skip2 = self.skip_layer2(fuse)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        branch0 = rgb
        branch1 = rgb + depth if self.fuse_depth_in_rgb_encoder == 'add' else self.se_layer3(rgb, depth)
        w = (weight[:, 0] + weight[:, 1] + weight[:, 2]).view(-1, 1, 1, 1)
        fuse = w * branch0 + (1 - w) * branch1
        skip3 = self.skip_layer3(fuse)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        branch0 = rgb
        branch1 = rgb + depth if self.fuse_depth_in_rgb_encoder == 'add' else self.se_layer4(rgb, depth)
        w = weight[:, 4].view(-1, 1, 1, 1)
        fuse = (1 - w) * branch0 + w * branch1
        out = self.context_module(fuse)
        out = self.decoder(enc_outs=[out, skip3, skip2, skip1])

        weight_mean = weight.mean(dim=0)
        loss = weight_mean * self.depth_enc_flop
        if test:
            if return_weight:
                return out, weight
            else:
                return out
        else:
            return out, loss.mean()

    def forward_flop(self, rgb, depth, w):
        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(depth)

        fuse = rgb + depth if self.fuse_depth_in_rgb_encoder == 'add' else self.se_layer0(rgb, depth)
        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # weight = self.gate_layer(rgb, depth, self.temp, self.hard_gate)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        if w < 1:
            fuse = rgb
        else:
            depth = self.encoder_depth.forward_layer1(depth)
            fuse = rgb + depth if self.fuse_depth_in_rgb_encoder == 'add' else self.se_layer1(rgb, depth)
        skip1 = self.skip_layer1(fuse)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        if w < 2:
            fuse = rgb
        else:
            depth = self.encoder_depth.forward_layer2(depth)
            fuse = rgb + depth if self.fuse_depth_in_rgb_encoder == 'add' else self.se_layer2(rgb, depth)
        skip2 = self.skip_layer2(fuse)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        if w < 3:
            fuse = rgb
        else:
            depth = self.encoder_depth.forward_layer3(depth)
            fuse = rgb + depth if self.fuse_depth_in_rgb_encoder == 'add' else self.se_layer3(rgb, depth)
        skip3 = self.skip_layer3(fuse)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        if w < 4:
            fuse = rgb
        else:
            depth = self.encoder_depth.forward_layer4(depth)
            fuse = rgb + depth if self.fuse_depth_in_rgb_encoder == 'add' else self.se_layer4(rgb, depth)

        # out = self.context_module(fuse)
        # out = self.decoder(enc_outs=[out, skip3, skip2, skip1])

        # return out


class GlobalGate(nn.Module):
    def __init__(self, branch_num, hidden_dim=8):
        super(GlobalGate, self).__init__()
        self.bnum = branch_num
        self.conv = nn.Sequential(
            nn.Conv2d(128, hidden_dim, kernel_size=5, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.Tanh(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.Tanh())
        self.fc = nn.Conv2d(hidden_dim, self.bnum, kernel_size=1, bias=False)

    def forward(self, rgb, depth, temp=1.0, hard=False):
        x = torch.concat([rgb, depth], dim=1)    # x shape (bs, 64*2, 120, 160)
        y = self.conv(x)
        y = F.adaptive_avg_pool2d(y, 1)
        y = self.fc(y)
        y = DiffSoftmax(y, tau=temp, hard=hard, dim=1)
        return y.squeeze(-1).squeeze(-1)


def main():
    height = 480
    width = 640

    model = SkipGateESANet(encoder_rgb='resnet50',
                           encoder_depth='resnet50',
                           encoder_block='BasicBlock',
                           fuse_depth_in_rgb_encoder='add').cuda()
    model.hard_gate = True

    # print(model)
    model.eval()
    rgb_image = torch.randn(1, 3, height, width).cuda()
    depth_image = torch.randn(1, 1, height, width).cuda()
    output = model(rgb_image, depth_image)

    # from thop import profile, clever_format
    # mac_list = []
    # for i in range(5):
    #     macs, params = profile(model, inputs=(rgb_image, depth_image, i))
    #     mac_list.append(macs / 1e9)
    # print(mac_list)
    # R34 + NBT1D
    # first depth encoder: 0.2506752
    # no decoder, no weight: [16.4731392 19.3337856 23.1694848 28.886784  31.761408 ]
    # no decoder: [16.59068459, 19.45133099, 23.28703019, 29.00432939, 31.87895339]
    # total, no weight: [22.2534697 25.1141161 28.9498153 34.6671145 37.5417385]
    # total: [22.37101509, 25.23166149, 29.06736069, 34.78465989, 37.65928389]

    # R50 + BasicBlock + SE-add fusion
    # first depth encoder:  0.2506752
    # no decoder: [26.635050552, 30.77858108, 37.108196504, 46.100204824, 51.063459352]
    # total: [32.5854654, 36.728995928, 43.058611352, 52.050619672, 57.0138742] 56.896328808


def see_gate_output():
    model = GlobalGate(2)
    rgb = torch.randn(6, 64, 120, 160)  # (64, 120, 160), (64, 120, 160), (128, 60, 80), (256, 30, 40)
    depth = torch.randn(6, 64, 120, 160)
    y = model(rgb, depth)
    # from thop import profile, clever_format
    # macs, params = profile(model, inputs=(rgb, depth,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(macs, params)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
