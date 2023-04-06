# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet import ResNet18, ResNet34, ResNet50
from src.models.rgb_depth_fusion import SqueezeAndExciteFusionAdd, SqueezeAndExciteReweigh
from src.models.context_modules import get_context_module
from src.models.resnet import BasicBlock, NonBottleneck1D
from src.models.model_utils import ConvBNAct, Swish, Hswish
from src.models.model import Decoder


class SkipESANet(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 encoder_rgb='resnet18',
                 encoder_depth='resnet18',
                 encoder_block='BasicBlock',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 pretrained_dir='./trained_models/imagenet',
                 activation='relu',
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 fuse_depth_in_rgb_encoder='SE-add',
                 upsampling='bilinear',
                 temp=1,
                 block_rule=None):  # 0: rgb only, 1: rgb+d, others: dynamic


        super(SkipESANet, self).__init__()

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

        # gating network & gumbel softmax
        self.temp = temp
        self.gate_layer0 = SqueezeAndExciteReweigh(self.temp, 64, activation=self.activation)
        self.gate_layer1 = SqueezeAndExciteReweigh(self.temp, self.encoder_rgb.down_4_channels_out, activation=self.activation)
        self.gate_layer2 = SqueezeAndExciteReweigh(self.temp, self.encoder_rgb.down_8_channels_out, activation=self.activation)
        self.gate_layer3 = SqueezeAndExciteReweigh(self.temp, self.encoder_rgb.down_16_channels_out, activation=self.activation)

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

        # statistics
        self.hard_gate = False
        self.ini_stage = False
        self.random_policy = False
        self.save_weight_info = False
        self.weight_list = [torch.Tensor() for _ in range(4)]

    def freeze(self):
        for name, param in self.named_parameters():
            if 'gate' not in name:
                param.requires_grad = False

    def start_weight(self):
        self.save_weight_info = True
        self.weight_list = [torch.Tensor() for _ in range(4)]

    def end_weight(self, print_each=False, thre=None):
        self.save_weight_info = False
        avg = []
        for i in range(4):
            if self.block_rule[i] != 2:
                continue
            if thre:
                print('-' * 40, 'layer ', i, '-' * 40)
                cnt1 = (self.weight_list[i][:, 0] < thre).sum()
                cnt2 = (self.weight_list[i][:, 1] < thre).sum()
                print(f'Skip {cnt1} branch 1 | {cnt2} branch 2')

            weight_mean = torch.mean(self.weight_list[i], axis=0)
            if print_each:
                print(self.weight_list[i])
                print(weight_mean)
            avg.append(weight_mean)
        self.weight_list = [torch.Tensor() for _ in range(4)]
        return avg


    def forward(self, rgb, depth, test=False):
        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(depth)

        fuse = rgb + depth
        weight0 = self.gate_layer0(rgb, depth, hard=self.hard_gate, random=self.random_policy, test=test)
        if self.save_weight_info:
            self.weight_list[0] = torch.cat((self.weight_list[0], weight0[:, :, 0, 0].cpu()))

        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)
        branch0, branch1 = rgb, rgb + depth

        prev_weight = None
        if self.block_rule[0] == 0:
            fuse = branch0
        elif self.block_rule[0] == 1:
            fuse = branch1
        else:
            fuse = weight0[:, 0:1, :, :] * branch0 + weight0[:, 1:2, :, :] * branch1
            prev_weight = weight0[:, 1, 0, 0] if not self.ini_stage else None
        weight1 = self.gate_layer1(rgb, depth, hard=self.hard_gate, prev_weight=prev_weight, random=self.random_policy, test=test)
        if self.save_weight_info:
            self.weight_list[1] = torch.cat((self.weight_list[1], weight1[:, :, 0, 0].cpu()))
        skip1 = self.skip_layer1(fuse)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        branch0, branch1 = rgb, rgb + depth

        if self.block_rule[1] == 0:
            fuse = branch0
        elif self.block_rule[1] == 1:
            fuse = branch1
        else:
            fuse = weight1[:, 0:1, :, :] * branch0 + weight1[:, 1:2, :, :] * branch1
            prev_weight = weight1[:, 1, 0, 0] if not self.ini_stage else None
        weight2 = self.gate_layer2(rgb, depth, hard=self.hard_gate, prev_weight=prev_weight, random=self.random_policy, test=test)
        if self.save_weight_info:
            self.weight_list[2] = torch.cat((self.weight_list[2], weight2[:, :, 0, 0].cpu()))
        skip2 = self.skip_layer2(fuse)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        branch0, branch1 = rgb, rgb + depth

        if self.block_rule[2] == 0:
            fuse = branch0
        elif self.block_rule[2] == 1:
            fuse = branch1
        else:
            fuse = weight2[:, 0:1, :, :] * branch0 + weight2[:, 1:2, :, :] * branch1
            prev_weight = weight2[:, 1, 0, 0] if not self.ini_stage else None
        weight3 = self.gate_layer3(rgb, depth, hard=self.hard_gate, prev_weight=prev_weight, random=self.random_policy, test=test)
        if self.save_weight_info:
            self.weight_list[3] = torch.cat((self.weight_list[3], weight3[:, :, 0, 0].cpu()))
        skip3 = self.skip_layer3(fuse)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        branch0, branch1 = rgb, rgb + depth

        if self.block_rule[3] == 0:
            fuse = branch0
        elif self.block_rule[3] == 1:
            fuse = branch1
        else:
            fuse = weight3[:, 0:1, :, :] * branch0 + weight3[:, 1:2, :, :] * branch1
            # print(weight[0, :, 0, 0])
        out = self.context_module(fuse)
        out = self.decoder(enc_outs=[out, skip3, skip2, skip1])

        # print('weight0', weight0[:, :, 0, 0])
        # print('weight1', weight1[:, :, 0, 0])
        # print('weight2', weight2[:, :, 0, 0])
        # print('weight3', weight3[:, :, 0, 0])

        return out


def main():
    height = 480
    width = 640

    model = SkipESANet(pretrained_on_imagenet=False, block_rule=[1, 1, 2, 2])
    model.hard_gate = True

    # print(model)

    model.eval()
    rgb_image = torch.randn(6, 3, height, width)
    depth_image = torch.randn(6, 1, height, width)

    with torch.no_grad():
        output = model(rgb_image, depth_image)
    print(output.shape)


if __name__ == '__main__':
    main()