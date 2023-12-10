import threading
import time
import argparse
from splitting import split_to_blocks
from rknn import random_knn
from tqdm import tqdm
import os

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
    parser.add_argument('--split', type=str, default='train',
                        help='split of the dataset')
    parser.add_argument('--blocks-limit', type=int, default=50,
                        help='Number of blocks to be processed at the same time')
                                                 
    params = parser.parse_args()
    
    points_datapath = datapath_list(params.data_dir, params.seq_ids, params.split)
    for file, filename in tqdm(enumerate(points_datapath), total=len(points_datapath)):
        point_cloud_blocks = split_to_blocks(filename, params.block_size, params.overlap, params.knn, params.knn) # change 9 according to the sequence number in the path
        length = len(point_cloud_blocks)
        b = 0 #block counter       
        while b < length:
            start = time.perf_counter()
            threads = []
            if length - b >= params.blocks_limit: 
                for pc in point_cloud_blocks[b:b+params.blocks_limit]:
                    t = threading.Thread(target=fixed_radius, args=[params.data_dir, pc, params.knn, filename.split("/")[9], b])
                    t.start()
                    threads.append(t)
                    b += 1
                    # print("block: ", b, "/", length," , Seq: ", filename.split("/")[9])
                j = 0 #job counter
                for thread in threads:
                    thread.join()
                    j += 1 
                finish = time.perf_counter()
                # print(f'Finished in {round(finish-start, 2)} seconds') 
            else:
                for pc in point_cloud_blocks[b:]:
                    t = threading.Thread(target=fixed_radius, args=[params.data_dir, pc, params.knn, filename.split("/")[9], b])
                    t.start()
                    threads.append(t)
                    b += 1
                    # print("block: ", b, "/", length," , Seq: ", filename.split("/")[9])
                j = 0
                for thread in threads:
                    thread.join()
                    j += 1
                finish = time.perf_counter()
                print(f'Finished in {round(finish-start, 2)} seconds') 
        del point_cloud_blocks
