import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm='G', skip=False, act='relu', drop=True,
                 bias=False):
        super(BasicBlock, self).__init__()
        self.skip = skip
        self.drop = drop
        self.in_channels = in_channels
        self.out_channels = out_channels
        n_groups = 8
        self.normconv = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                                  padding=1, bias=bias)
        self.dilaconv = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                                  dilation=2, padding=2, bias=bias)
        self.fuseconv = nn.Conv3d(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=0,
                                  bias=bias)
        self.dropout = nn.Dropout3d(0.6, inplace=True)
        if norm == 'G':
            self.normlize = nn.GroupNorm(n_groups, self.in_channels)
        else:
            self.normlize = nn.InstanceNorm3d(self.in_channels)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.normlize(x)
        out = self.act(out)
        normx = self.normconv(out)
        dilax = self.dilaconv(out)
        addx = self.act(normx + dilax)
        mulx = self.act(normx * dilax)
        out = torch.cat([mulx, addx], 1)
        out = self.fuseconv(out)
        if self.drop:
            out = self.dropout(out)
        if self.skip:
            out = out + x
        return out


#
# class BasicBlockV2(nn.Module):
#     def __init__(self, in_channels, out_channels, norm='G', skip=False, inside_skip='ADD', act='relu', drop=True,
#                  bias=False):
#         super(BasicBlockV2, self).__init__()
#         self.skip = skip
#         self.drop = drop
#         self.inside_skip = inside_skip
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.dropout = nn.Dropout3d(p=0.5, inplace=True)
#         n_groups = 8
#
#         self.basicconv1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
#                                     stride=1,
#                                     padding=1, bias=bias)
#         self.normconv1 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3,
#                                    stride=1,
#                                    padding=1, bias=bias)
#         self.dilaconv1 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3,
#                                    stride=1,
#                                    dilation=2, padding=2, bias=bias)
#         if self.inside_skip == 'CAT':
#             self.fuseconv1 = nn.Conv3d(3 * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1,
#                                        bias=bias)
#         else:
#             self.fuseconv1 = nn.Conv3d(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1,
#                                        bias=bias)
#
#         if norm == 'G':
#             self.normlize1 = nn.GroupNorm(n_groups, self.out_channels)
#             self.normlize2 = nn.GroupNorm(n_groups, self.out_channels)
#
#         else:
#             self.normlize1 = nn.InstanceNorm3d(self.out_channels)
#             self.normlize2 = nn.InstanceNorm3d(self.out_channels)
#
#         if act == 'relu':
#             self.act = nn.ReLU(inplace=True)
#         else:
#             self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#         self.basicconv2 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3,
#                                     stride=1,
#                                     padding=1, bias=bias)
#         self.normconv2 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3,
#                                    stride=1,
#                                    padding=1, bias=bias)
#         self.dilaconv2 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3,
#                                    stride=1,
#                                    dilation=2, padding=2, bias=bias)
#         if self.inside_skip == 'CAT':
#             self.fuseconv2 = nn.Conv3d(3 * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1,
#                                        bias=bias)
#         else:
#             self.fuseconv2 = nn.Conv3d(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1,
#                                        bias=bias)
#
#     def forward(self, x):
#
#         out = self.normlize1(x)  # B
#         out = self.act(out)  # R
#         out = self.basicconv1(out)  # c
#         basic1_out = out
#         if self.drop:
#             out = self.dropout(out)
#         normx1 = self.normconv1(out)
#         dilax1 = self.dilaconv1(out)
#         addx1 = self.act(normx1 + dilax1)
#         mulx1 = self.act(normx1 * dilax1)
#         out = torch.cat([mulx1, addx1], 1)
#         if self.inside_skip == 'CAT':
#             out = torch.cat([out, basic1_out], 1)
#             out = self.fuseconv1(out)
#         if self.inside_skip == 'ADD':
#             out = self.fuseconv1(out)
#             out = out + basic1_out
#         out = self.normlize2(out)  # B
#         out = self.act(out)  # R
#         out = self.basicconv2(out)  # c
#         basic2_out = out
#         if self.drop:
#             out = self.dropout(out)
#         normx2 = self.normconv2(out)
#         dilax2 = self.dilaconv2(out)
#         addx2 = self.act(normx2 + dilax2)
#         mulx2 = self.act(normx2 * dilax2)
#         out = torch.cat([mulx2, addx2], 1)
#         if self.inside_skip == 'CAT':
#             out = torch.cat([out, basic1_out], 1)
#             out = self.fuseconv1(out)
#         if self.inside_skip == 'ADD':
#             out = self.fuseconv1(out)
#             out = out + basic2_out
#         if self.skip:
#             out = out + x
#         return out


##add 5*5conv to
class BasicBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, norm='G', skip=True, inside_skip='CAT', act='relu', drop=True,
                 bias=False):
        super(BasicBlockV2, self).__init__()
        self.skip = skip
        self.drop = drop
        self.inside_skip = inside_skip
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout3d(p=0.2, inplace=True)
        n_groups = 8

        if norm == 'G':
            self.normlize1 = nn.GroupNorm(n_groups, self.out_channels)
            self.normlize2 = nn.GroupNorm(n_groups, self.out_channels)

        else:
            self.normlize1 = nn.InstanceNorm3d(self.out_channels)
            self.normlize2 = nn.InstanceNorm3d(self.out_channels)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.normconv5_1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=5,
                                     stride=1, padding=2, bias=bias)
        self.normconv3_1 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3,
                                     stride=1, padding=1, bias=bias)
        self.dilaconv1 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3,
                                   stride=1, dilation=2, padding=2, bias=bias)

        self.normconv5_2 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=5,
                                     stride=1, padding=2, bias=bias)
        self.normconv3_2 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3,
                                     stride=1, padding=1, bias=bias)
        self.dilaconv2 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3,
                                   stride=1, dilation=2, padding=2, bias=bias)

        if self.inside_skip == 'CAT':
            self.fuseconv1 = nn.Conv3d(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1,
                                       bias=bias)
            self.fuseconv2 = nn.Conv3d(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1,
                                       bias=bias)

    def forward(self, x):

        out = self.normlize1(x)  # B
        out = self.act(out)  # R
        norm5_1 = self.normconv5_1(out)
        norm3_1 = self.normconv3_1(out)
        dila3_1 = self.dilaconv1(out)
        addx1 = self.act(norm5_1 + norm3_1 + dila3_1)
        mult1 = self.act(norm5_1 * norm3_1 * dila3_1)
        if self.inside_skip == 'CAT':
            temp = torch.cat([addx1, mult1], 1)
            out = self.fuseconv1(temp)
        else:
            out = addx1 + mult1

        out = self.dropout(out)

        # second block
        out = self.normlize1(out)  # B
        out = self.act(out)  # R
        norm5_2 = self.normconv5_2(out)
        norm3_2 = self.normconv3_2(out)
        dila3_2 = self.dilaconv2(out)
        addx2 = self.act(norm5_2 + norm3_2 + dila3_2)
        mult2 = self.act(norm5_2 * norm3_2 * dila3_2)
        if self.inside_skip == 'CAT':
            temp = torch.cat([addx2, mult2], 1)
            out = self.fuseconv1(temp)
        else:
            out = addx2 + mult2

        if self.skip:
            out = out + x
        return out


class HDCUnet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, init_channels=32, bias=False, skip=False, norm='G', act='relu',
                 sig=True, gtcount=False):
        super(HDCUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.sig = sig
        self.mode = 'trilinear'
        self.gtcount = gtcount

        self.initconv = nn.Conv3d(in_channels, init_channels, kernel_size=3, padding=1, stride=1, bias=bias)

        ###encoder start
        self.level1 = nn.Sequential(BasicBlock(init_channels, init_channels, norm=norm, skip=skip, act=act),
                                    BasicBlock(init_channels, init_channels, norm=norm, skip=skip, act=act))
        self.level1_down = nn.Conv3d(init_channels, 2 * init_channels, kernel_size=3, padding=1, stride=2, bias=bias)

        self.level2 = nn.Sequential(BasicBlock(2 * init_channels, 2 * init_channels, norm=norm, skip=skip, act=act),
                                    BasicBlock(2 * init_channels, 2 * init_channels, norm=norm, skip=skip, act=act))
        self.level2_down = nn.Conv3d(2 * init_channels, 4 * init_channels, kernel_size=3, padding=1, stride=2,
                                     bias=bias)

        self.level3 = nn.Sequential(BasicBlock(4 * init_channels, 4 * init_channels, norm=norm, skip=skip, act=act),
                                    BasicBlock(4 * init_channels, 4 * init_channels, norm=norm, skip=skip, act=act),
                                    BasicBlock(4 * init_channels, 4 * init_channels, norm=norm, skip=skip, act=act))
        self.level3_down = nn.Conv3d(4 * init_channels, 8 * init_channels, kernel_size=3, padding=1, stride=2,
                                     bias=bias)

        self.level4 = nn.Sequential(BasicBlock(8 * init_channels, 8 * init_channels, norm=norm, skip=skip, act=act),
                                    BasicBlock(8 * init_channels, 8 * init_channels, norm=norm, skip=skip, act=act),
                                    BasicBlock(8 * init_channels, 8 * init_channels, norm=norm, skip=skip, act=act))
        self.level4_down = nn.Conv3d(8 * init_channels, 16 * init_channels, kernel_size=3, padding=1, stride=2,
                                     bias=bias)
        ##encoder stop

        self.flat = nn.Sequential(BasicBlock(16 * init_channels, 16 * init_channels, norm=norm, skip=skip, act=act),
                                  BasicBlock(16 * init_channels, 16 * init_channels, norm=norm, skip=skip, act=act),
                                  BasicBlock(16 * init_channels, 16 * init_channels, norm=norm, skip=skip, act=act))

        # decoder start
        self.level4_up_conv = nn.Conv3d(16 * init_channels, 8 * init_channels, kernel_size=3, padding=1, stride=1,
                                        bias=bias)
        self.level4_up = nn.Sequential(
            BasicBlock(8 * init_channels, 8 * init_channels, norm=norm, skip=skip, act=act),
            BasicBlock(8 * init_channels, 8 * init_channels, norm=norm, skip=skip, act=act),
            BasicBlock(8 * init_channels, 8 * init_channels, norm=norm, skip=skip, act=act))

        self.level3_up_conv = nn.Conv3d(8 * init_channels, 4 * init_channels, kernel_size=3, padding=1, stride=1,
                                        bias=bias)
        self.level3_up = nn.Sequential(BasicBlock(4 * init_channels, 4 * init_channels, norm=norm, skip=skip, act=act),
                                       BasicBlock(4 * init_channels, 4 * init_channels, norm=norm, skip=skip, act=act),
                                       BasicBlock(4 * init_channels, 4 * init_channels, norm=norm, skip=skip, act=act))

        self.level2_up_conv = nn.Conv3d(4 * init_channels, 2 * init_channels, kernel_size=3, padding=1, stride=1,
                                        bias=bias)
        self.level2_up = nn.Sequential(BasicBlock(2 * init_channels, 2 * init_channels, norm=norm, skip=skip, act=act),
                                       BasicBlock(2 * init_channels, 2 * init_channels, norm=norm, skip=skip, act=act))

        self.level1_up_conv = nn.Conv3d(2 * init_channels, init_channels, kernel_size=3, padding=1, stride=1,
                                        bias=bias)
        self.level1_up = nn.Sequential(BasicBlock(init_channels, init_channels, norm=norm, skip=skip, act=act),
                                       BasicBlock(init_channels, init_channels, norm=norm, skip=skip, act=act))

        self.finalconv = nn.Conv3d(init_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)

        if self.gtcount:
            self.gp = nn.AdaptiveAvgPool3d(2)
            self.fc1 = nn.Linear(2048, 4096)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(4096, 512)

        if self.sig:
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Softmax(1)

    def forward(self, x):
        x = self.initconv(x)

        l1x = self.level1(x)
        l1x_down = self.level1_down(l1x)

        l2x = self.level2(l1x_down)
        l2x_down = self.level2_down(l2x)

        l3x = self.level3(l2x_down)
        l3x_down = self.level3_down(l3x)

        l4x = self.level4(l3x_down)
        l4x_down = self.level4_down(l4x)

        l4x_down = self.flat(l4x_down)

        out = F.interpolate(l4x_down, l4x.shape[2:], mode=self.mode)
        out = self.level4_up_conv(out)
        out = out + l4x
        out = self.level4_up(out)

        out = F.interpolate(out, l3x.shape[2:], mode=self.mode)
        out = self.level3_up_conv(out)
        out = out + l3x
        out = self.level3_up(out)

        out = F.interpolate(out, l2x.shape[2:], mode=self.mode)
        out = self.level2_up_conv(out)
        out = out + l2x
        out = self.level2_up(out)

        out = F.interpolate(out, l1x.shape[2:], mode=self.mode)
        out = self.level1_up_conv(out)
        out = out + l1x
        out = self.level1_up(out)

        out = self.finalconv(out)
        out = self.final_act(out)

        if self.gtcount:
            l4x_down = self.gp(l4x_down)
            l4x_down = l4x_down.view(l4x_down.shape[0], -1)
            l4x_down = self.fc1(l4x_down)
            l4x_down = self.dropout(l4x_down)
            l4x_down = self.fc2(l4x_down)
            return out, l4x_down
        else:
            return out


class HDCUnetV2(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, init_channels=16, bias=False, skip=True, norm='G', act='relu',
                 sig=True, gtcount=True, inside_skip=None):
        super(HDCUnetV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.sig = sig
        self.mode = 'trilinear'
        self.gtcount = gtcount

        self.initconv = nn.Sequential(
            nn.Conv3d(in_channels, init_channels, kernel_size=3, padding=1, stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(init_channels, init_channels, kernel_size=3, padding=1,
                      stride=1, bias=bias))

        ###encoder start
        self.level1 = BasicBlockV2(init_channels, init_channels, norm=norm, skip=skip, act=act, inside_skip=inside_skip)
        self.level1_down = nn.Conv3d(init_channels, 2 * init_channels, kernel_size=3, padding=1, stride=2, bias=bias)

        self.level2 = BasicBlockV2(2 * init_channels, 2 * init_channels, norm=norm, skip=skip, act=act,
                                   inside_skip=inside_skip)
        self.level2_down = nn.Conv3d(2 * init_channels, 4 * init_channels, kernel_size=3, padding=1, stride=2,
                                     bias=bias)

        self.level3 = BasicBlockV2(4 * init_channels, 4 * init_channels, norm=norm, skip=skip, act=act,
                                   inside_skip=inside_skip)
        self.level3_down = nn.Conv3d(4 * init_channels, 8 * init_channels, kernel_size=3, padding=1, stride=2,
                                     bias=bias)

        self.level4 = BasicBlockV2(8 * init_channels, 8 * init_channels, norm=norm, skip=skip, act=act,
                                   inside_skip=inside_skip)
        self.level4_down = nn.Conv3d(8 * init_channels, 16 * init_channels, kernel_size=3, padding=1, stride=2,
                                     bias=bias)
        ##encoder stop

        self.flat = BasicBlockV2(16 * init_channels, 16 * init_channels, norm=norm, skip=skip, act=act,
                                 inside_skip=inside_skip)
        # decoder start
        self.level4_up_conv = nn.Conv3d(16 * init_channels, 8 * init_channels, kernel_size=3, padding=1, stride=1,
                                        bias=bias)
        self.level4_up = BasicBlockV2(8 * init_channels, 8 * init_channels, norm=norm, skip=skip, act=act,
                                      inside_skip=inside_skip)

        self.level3_up_conv = nn.Conv3d(8 * init_channels, 4 * init_channels, kernel_size=3, padding=1, stride=1,
                                        bias=bias)
        self.level3_up = BasicBlockV2(4 * init_channels, 4 * init_channels, norm=norm, skip=skip, act=act,
                                      inside_skip=inside_skip)

        self.level2_up_conv = nn.Conv3d(4 * init_channels, 2 * init_channels, kernel_size=3, padding=1, stride=1,
                                        bias=bias)
        self.level2_up = BasicBlockV2(2 * init_channels, 2 * init_channels, norm=norm, skip=skip, act=act,
                                      inside_skip=inside_skip)

        self.level1_up_conv = nn.Conv3d(2 * init_channels, init_channels, kernel_size=3, padding=1, stride=1,
                                        bias=bias)
        self.level1_up = BasicBlockV2(init_channels, init_channels, norm=norm, skip=skip, act=act,
                                      inside_skip=inside_skip)

        self.finalconv = nn.Conv3d(init_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)

        if self.gtcount:
            self.gp = nn.AdaptiveAvgPool3d(2)
            self.fc1 = nn.Linear(2048, 4096)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(4096, 512)

        if self.sig:
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Softmax(1)

    def forward(self, x):
        x = self.initconv(x)
        l1x = self.level1(x)
        l1x_down = self.level1_down(l1x)

        l2x = self.level2(l1x_down)
        l2x_down = self.level2_down(l2x)

        l3x = self.level3(l2x_down)
        l3x_down = self.level3_down(l3x)

        l4x = self.level4(l3x_down)
        l4x_down = self.level4_down(l4x)

        l4x_down = self.flat(l4x_down)
        out = F.interpolate(l4x_down, l4x.shape[2:], mode=self.mode)
        out = self.level4_up_conv(out)
        out = out + l4x
        out = self.level4_up(out)

        out = F.interpolate(out, l3x.shape[2:], mode=self.mode)
        out = self.level3_up_conv(out)
        out = out + l3x
        out = self.level3_up(out)

        out = F.interpolate(out, l2x.shape[2:], mode=self.mode)
        out = self.level2_up_conv(out)
        out = out + l2x
        out = self.level2_up(out)

        out = F.interpolate(out, l1x.shape[2:], mode=self.mode)
        out = self.level1_up_conv(out)
        out = out + l1x
        out = self.level1_up(out)

        out = self.finalconv(out)
        out = self.final_act(out)

        if self.gtcount:
            l4x_down = self.gp(l4x_down)
            l4x_down = l4x_down.view(l4x_down.shape[0], -1)
            l4x_down = self.fc1(l4x_down)
            l4x_down = self.dropout(l4x_down)
            l4x_down = self.fc2(l4x_down)
            return out, l4x_down
        else:
            return out
