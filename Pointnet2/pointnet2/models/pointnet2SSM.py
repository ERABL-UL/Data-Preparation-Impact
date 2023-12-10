# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from model.pointnet2.pointnet2_modules import PointnetSAModuleMSGVotes, PointnetFPModule
from model.pointnet2.pointnet2_utils import furthest_point_sample

import MinkowskiEngine as ME


class PointNet2SemSegMSG(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, num_classes, num_feats):
        super().__init__()
        self.sa1 = PointnetSAModuleMSGVotes(
                npoint=1024,
                radii= [0.05, 0.1],
                nsamples= [16, 32],
                mlps= [[0, 16, 16, 32],[0, 32, 32, 64]],
                use_xyz=True
            )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.sa2 = PointnetSAModuleMSGVotes(
                npoint=256,
                radii= [0.1, 0.2],
                nsamples= [16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=True
            )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.sa3 = PointnetSAModuleMSGVotes(
                npoint=64,
                radii= [0.2, 0.4],
                nsamples= [16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=True
            )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.sa4 = PointnetSAModuleMSGVotes(
                npoint=16,
                radii= [0.4, 0.8],
                nsamples= [16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=True
            )
        c_out_3 = 512 + 512
        
        self.fp1 = PointnetFPModule(mlp=[256 + 0, 128, 128])
        self.fp2 = PointnetFPModule(mlp=[512 + c_out_0, 256, 256])
        self.fp3 = PointnetFPModule(mlp=[512 + c_out_1, 512, 512])
        self.fp4 = PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512])
        
        self.fc = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 20, kernel_size=1),
        )
        # self.conv1 = nn.Conv1d(128, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, 20, 1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """

        xyz0, features0 = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz1, features1, _ = self.sa1(xyz0, features0)
        xyz2, features2, _ = self.sa2(xyz1, features1)
        xyz3, features3, _ = self.sa3(xyz2, features2)
        xyz4, features4, _ = self.sa4(xyz3, features3)

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features3 = self.fp4(xyz3, xyz4, features3, features4)
        features2 = self.fp3(xyz2, xyz3, features2, features3)
        features1 = self.fp2(xyz1, xyz2, features1, features2)
        features0 = self.fp1(xyz0, xyz1, features0, features1) 
        features = self.fc(features0)
        
        # x = self.drop1(F.relu(self.bn1(self.conv1(features0))))
        # x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)

        return features
