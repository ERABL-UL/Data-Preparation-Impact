import argparse
import time
import os
from splitting import split_to_blocks
from db_fps import density_based_fps
from tqdm import tqdm

def datapath_list(root, seq_ids, split):
    points_datapath = []
    for seq in seq_ids:
        point_seq_path = os.path.join(root, split, 'sequences', seq)
        point_seq_ply = os.listdir(point_seq_path)
        point_seq_ply.sort()
        points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_ply ]
    return points_datapath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')
    parser.add_argument('--data-dir', type=str, default='/home/reza/PHD/Data/KITTI360',
                        help='Path to dataset (default: ./Datasets/SemanticKITTI')
    parser.add_argument('--seq-ids', type=list, default=['00','02','03','04','05','06','07','09','10'] ,
                        help='list of sequences #numbers')
    parser.add_argument('--block-size', type=int, default=20,
                        help='Size of the blocks to partition the point cloud')
    parser.add_argument('--overlap', type=int, default=10,
                        help='Overlap between the blocks')
    parser.add_argument('--knn', type=int, default=8192,
                        help='number of points in each box')
    parser.add_argument('--ds-rate', type=int, default=30,
                        help='The downsampling rate')
    parser.add_argument('--high-d', type=int, default=70,
                        help='high density rate')        
    parser.add_argument('--low-d', type=int, default=30,
                        help='low density rate')        
    parser.add_argument('--split', type=str, default='train',
                        help='split of the dataset')
    parser.add_argument('--block-limit', type=int, default=20000,
                        help='TNumber of points to extraxt the boxes from')    
                                                 
    params = parser.parse_args()
    
    points_datapath = datapath_list(params.data_dir, params.seq_ids, params.split)
    for file, filename in tqdm(enumerate(points_datapath), total=len(points_datapath)):
        point_cloud_blocks = split_to_blocks(filename, params.block_size, params.overlap,
                                             params.knn, params.block_limit) # change 9 according to the sequence number in the path
        for b, pc in enumerate(point_cloud_blocks):
            density_based_fps(pc, params.data_dir, params.ds_rate, params.high_d, 
                              params.low_d, params.knn, b, filename.split("/")[9])

