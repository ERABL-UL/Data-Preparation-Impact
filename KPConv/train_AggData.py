import signal
import os
import numpy as np
import sys
import torch

# Dataset
from data_utils.DataLoader import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SparseSimCLR')
    parser.add_argument('--dataset', type=str, default='AggregatedDataset',
                        help='Name of the dataset')
    parser.add_argument('--data-dir', type=str, default='/home/reza/PHD/Data/KITTI360/fps_knn',
                        help='Path to dataset (default: ./Datasets/KITTI360')
    parser.add_argument('--dataset-task', type=str, default='agg_segmentation',
                        help='task to be done for this dataset (default: agg_segmentation')
    parser.add_argument('--saving', default=True,
                        help='If you want to save the updated args')
    parser.add_argument('--saving-path', default= None,
                        help='Saving path for the training logs')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--max-epoch', type=int, default=10,
                        help='number of training epochs (default: 10)')
    parser.add_argument('--epoch-steps', type=int, default=500,
                        help='steps for each epoch (default: 500)')
    parser.add_argument('--validation-size', type=int, default=200,
                        help='number of validation epoch steps (default: 200)')
    parser.add_argument('--batch-num', type=int, default=8,
                        help='input training batch-size')
    parser.add_argument('--max-in-points', type=int, default=100000,
                        help='max number of input points')
    parser.add_argument('--in-radius', type=int, default=10,
                        help='radius of the spheres')
    parser.add_argument('--val-batch-num', type=int, default=8,
                        help='input validation batch-size')
    parser.add_argument('--max_val_points', type=int, default=100000,
                        help='max number of input points for validation')
    parser.add_argument('--val-radius', type=int, default=10,
                        help='radius of the spheres for validation')   
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='learning rate (default: 1e-3')
    parser.add_argument("--lr-decays", default={},
                        action="store", type=dict, help='lr decay (default: 1e-3')
    parser.add_argument("--momentum", default=0.9, type=float,
                        help='Learning rate momentum (default: 0.9')
    parser.add_argument("--weight-decay", default=1e-3, type=float,
                        help='weight decay (default: 0.9')
    parser.add_argument('--model-checkpoint', type=str, default='checkpoint',
                        help='checkpoint directory (default: checkpoint)')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                        help='using gpu (default: False')
    parser.add_argument('--batch-limit', type=int, default=6*8192,
                        help='Number of points sampled from point clouds (default: 8192')
    parser.add_argument('--num-classes', type=int, default=7,
                        help='number of classes')
    parser.add_argument('--ignored-labels', type=int, default=0,
                        help='list of labels to be ignored')
    parser.add_argument('--input-threads', type=int, default=0,
                        help='Number of CPU threads for the input pipeline')
    parser.add_argument('--first-subsampling-dl', type=float, default=0.015,
                        help='the size of the first subsampling layer')
    parser.add_argument('--augment-color', type=float, default=0.8,
                        help='Augmentation parameters')
    parser.add_argument('--in-features-dim', type=int, default=4,
                        help='Number of CPU threads for the input pipeline')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of layers of the network (will be updated)')
    parser.add_argument('--augment-rotation', type=str, default='vertical',
                        help='Augmentation parameters')
    parser.add_argument('--augment-scale-min', type=float, default=0.9,
                        help='Augmentation parameters')
    parser.add_argument('--augment-scale-max', type=float, default=1.1,
                        help='Augmentation parameters')
    parser.add_argument('--augment-scale-anisotropic', default=True,
                        help='Augmentation parameters')
    parser.add_argument('--augment-symmetries', type=list, default=[False, False, False],
                        help='Augmentation parameters')
    parser.add_argument('--augment-noise', type=float, default=0.005,
                        help='Augmentation parameters')
    parser.add_argument('--conv-radius', type=float, default=2.5,
                        help='Radius of convolution in "number grid cell". (2.5 is the standard value)')
    parser.add_argument('--deform-layers', type=list, default=[],
                        help='deformable layers, will be updated')
    parser.add_argument('--deform-radius', type=float, default=5,
                        help='Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out')
    parser.add_argument('--num-kernel-points', type=int, default=15,
                        help='Number of kernel points')
    parser.add_argument('--class-w', type=list, default=[],
                        help='class weights')
    parser.add_argument('--batch-norm-momentum', type=float, default= 0.02,
                        help='Batch normalization parameters')
    parser.add_argument('--use-batch-norm', default=True,
                        help='Batch normalization parameters')
    parser.add_argument('--KP-extent', type=float, default= 1.2,
                        help='Kernel point influence radius')
    parser.add_argument('--in-points-dim', type=int, default= 3,
                        help='Dimension of input points')
    parser.add_argument('--KP-influence', type=str, default='linear',
                        help='Influence function when d < KP_extent. (constant, linear, gaussian) When d > KP_extent, always zero')
    parser.add_argument('--aggregation-mode', type=str, default='sum',
                        help='Aggregation function of KPConv in (closest, sum). Decide if you sum all kernel point influences, or if you only take the influence of the closest KP')
    parser.add_argument('--modulated', default=False,
                        help='Use modulateion in deformable convolutions')
    parser.add_argument('--fixed-kernel-points', type=str, default='center',
                        help='Fixed points in the kernel : none, center or verticals')
    parser.add_argument('--first-features-dim', type=int, default=128,
                        help='Dimension of the first feature maps')
    parser.add_argument('--deform-fitting-mode', type=str, default='point2point',
                        help='Deformable offset loss, point2point fitting geometry by penalizing distance from deform point to input points')
    parser.add_argument('--deform-fitting-power', type=float, default=1.0,
                        help='Multiplier for the fitting/repulsive loss')
    parser.add_argument('--deform-lr-factor', type=float, default= 0.1,
                        help='Multiplier for learning rate applied to the deformations')
    parser.add_argument('--repulse-extent', type=float, default= 1.2,
                        help='Distance of repulsion for deformed kernel points')
    parser.add_argument('--grad-clip-norm', type=float, default= 100.0,
                        help='Gradient clipping value (negative means no clipping)')
    parser.add_argument('--checkpoint-gap', type=int, default= 50,
                        help='Number of epoch between each checkpoint')
    parser.add_argument('--equivar-mode', type=str, default='',
                        help='Decide the mode of equivariance and invariance')
    parser.add_argument('--invar-mode', type=str, default='',
                        help='Decide the mode of equivariance and invariance')
    parser.add_argument('--segmentation-ratio', type=float, default= 1.0,
                        help='For segmentation models : ratio between the segmented area and the input area')
    parser.add_argument('--n-frames', type=int, default= 1,
                        help='Number of frames to merge')
    parser.add_argument('--augment-occlusion', type=str, default='none',
                        help='Augmentation parameters')
    parser.add_argument('--augment-occlusion-ratio', type=float, default=0.2,
                        help='Augmentation parameters')
    parser.add_argument('--augment-occlusion-num', type=int, default=1,
                        help='Augmentation parameters')
    parser.add_argument('--segloss-balance', type=str, default='none',
                        help='The way we balance segmentation loss DEPRECATED')
    parser.add_argument('--architecture', type=list, default=['simple', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb',
                    'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided',
                    'resnetb', 'resnetb', 'resnetb_strided', 'resnetb', 'nearest_upsample',
                    'unary', 'nearest_upsample', 'unary', 'nearest_upsample', 'unary',
                    'nearest_upsample', 'unary'],
                        help='definition of architecture layers')
    
      
    args = parser.parse_args()
    args.num_layers = len([block for block in args.architecture if 'pool' in block or 'strided' in block]) + 1
    args.lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, args.max_epoch)}

    # Initialize datasets

    training_dataset = AggDataset(args, "train", balance_classes=True)
    test_dataset = AggDataset(args, 'validation', balance_classes=False)

    # Initialize samplers
    training_sampler = AggSampler(training_dataset, args)
    test_sampler = AggSampler(test_dataset, args)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=AggCollate,
                                 num_workers=args.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=AggCollate,
                             num_workers=args.input_threads,
                             pin_memory=True)
    # Calibrate max_in_point value
    training_sampler.calib_max_in(training_loader, verbose=True)
    test_sampler.calib_max_in(test_loader, verbose=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)


    net = KPFCNN(args, training_dataset.label_values, training_dataset.ignored_labels)

    # Choose here if you want to start training from a previous snapshot (None for new training)
    # previous_training_path = 'Log_2020-03-19_19-53-27'
    previous_training_path = ''

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    # Define a trainer class
    trainer = ModelTrainer(net, args, chkp_path=chosen_chkp)

    # Training
    trainer.train(net, training_loader, test_loader, args)

