import torch
from data_utils.DataLoaderAgg import *
from torch.utils.data import DataLoader
from train_lib.trainer import training
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')

    parser.add_argument('--data-dir', type=str, default='/home/reza/PHD/Data/KITTI360/rknn',
                        help='Path to dataset (default: ./Datasets/KITTI360')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of training epochs (default: 200)')
    parser.add_argument('--model-name', type=str, default='pointnet2_msg_sem',
                        help='model name to load (default: pointnet2_msg_sem)')    
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3')
    parser.add_argument("--lr-decays", default={i: 0.1 ** (1 / 150) for i in range(1, 10)},
                        action="store", type=dict, help='lr decay (default: 1e-3')
    parser.add_argument("--momentum", default=0.9, action="store", type=float,
                        help='Learning rate momentum (default: 0.9')
    parser.add_argument('--model-checkpoint', type=str, default='checkpoint',
                        help='checkpoint directory (default: checkpoint)')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                        help='using gpu (default: False')
    parser.add_argument('--in_feats', type=int, default=0,
                        help='Feature input size (default: 0')
    parser.add_argument('--batch-limit', type=int, default=8192,
                        help='Number of points sampled from point clouds (default: 8192')
    parser.add_argument('--num-classes', type=int, default=7,
                        help='number of classes')
    parser.add_argument('--ignored-labels', type=int, default=0,
                        help='list of labels to be ignored')
    parser.add_argument('--input-threads', type=int, default=8,
                        help='Number of CPU threads for the input pipeline')
    args = parser.parse_args()
    args.lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, args.epochs)}
    training_dataset = AggDataset(args, split="train")
    validation_dataset = AggDataset(args, split="validation")
    
    # Initialize samplers
    training_sampler = SemanticKittiAggSampler(training_dataset, args)
    validation_sampler = SemanticKittiAggSampler(validation_dataset, args)
    
    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=args.batch_size,
                                 sampler=training_sampler,
                                 collate_fn=SemanticKittiAggCollate,
                                 num_workers=args.input_threads,
                                 pin_memory=True)
    validation_loader = DataLoader(validation_dataset,
                                 batch_size=args.batch_size,
                                 sampler=validation_sampler,
                                 collate_fn=SemanticKittiAggCollate,
                                 num_workers=args.input_threads,
                                 pin_memory=True)
    loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignored_labels)
    training(args, training_loader, validation_loader, loss)

