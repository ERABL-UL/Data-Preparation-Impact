import numpy as np
import torch
from data_utils.DataLoaderAgg import *
from torch.utils.data import DataLoader
import argparse
import importlib
from tqdm import tqdm
from train_lib.eval_utils import *
from data_utils.data_map import labels_poss as labels
import warnings
warnings.filterwarnings("ignore")

def test(params, model, criterion, test_loader):
    model.eval()
    losses = []
    accuracies = []
    ious = []
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Test", unit="batch") as t:
            for i , batch in enumerate(test_loader):
                if len(batch.points) > 1:
                    if params.use_gpu:
                        batch.to(torch.device("cuda:0"))                
                    outputs = model(torch.stack(batch.points).float())
                    loss = criterion(outputs, batch.labels.squeeze().long())
                    losses.append(loss.cpu().item())
                    accuracies.append(accuracy(outputs, batch.labels.squeeze().long()))
                    ious.append(intersection_over_union(outputs, batch.labels.squeeze().long()))
                    t.set_postfix(accuracy=np.nanmean(np.array(ious), axis=0)[-1], loss=np.mean(losses))
                    t.update()
    return np.mean(losses), np.nanmean(np.array(accuracies), axis=0), np.nanmean(np.array(ious), axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')

    parser.add_argument('--data-dir', type=str, default='/home/reza/PHD/Data/KITTI360/rknn',
                        help='Path to dataset (default: ./Datasets/KITTI360')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--trained-epoch', type=str, default="9",
                        help='name of the pre-trained checkpoint')
    parser.add_argument('--model-name', type=str, default='pointnet2_msg_sem',
                        help='model name to load (default: pointnet2_msg_sem)')    
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

    test_dataset = AggDataset(args, split="validation")
    test_sampler = SemanticKittiAggSampler(test_dataset, args)
    test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 sampler=test_sampler,
                                 collate_fn=SemanticKittiAggCollate,
                                 num_workers=args.input_threads,
                                 pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignored_labels)
    _model = importlib.import_module(args.model_name)
    model = _model.PointNet2SemSegMSG(args).cuda()
    file_name = f'./{args.model_checkpoint}/checkpoint_{args.trained_epoch}.pt'
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint)
    print(f'Model {file_name} loaded from epoch {args.trained_epoch}')
    loss, accs, ious = test(args, model, criterion, test_loader)
    
    
