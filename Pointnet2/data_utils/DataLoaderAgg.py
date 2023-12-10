import numpy as np
import pickle
import torch
import OSToolBox as ost

import os
from os import listdir
from os.path import exists, join

from data_utils import data_map
# from data_utils.common import PointCloudDataset
from torch.utils.data import Sampler


class AggDataset():

    def __init__(self, params, split):
        # Dataset folder
        self.params = params
        self.path = self.params.data_dir
        self.split = split
        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array(self.params.ignored_labels)
        # Read labels
        all_labels = data_map.labels
        learning_map_inv = data_map.learning_map_inv
        learning_map = data_map.learning_map
        self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
        for k, v in learning_map.items():
            self.learning_map[k] = v
        self.learning_map_inv = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32)
        for k, v in learning_map_inv.items():
            self.learning_map_inv[k] = v
        
        # Dict from labels to names
        self.label_to_names = {k: all_labels[v] for k, v in learning_map_inv.items()}
        self.num_classes = self.params.num_classes
        # Training or test set
        self.seq_ids = {} #change to the desire sequences for your dataset
        self.seq_ids['train'] = [ '00', '02', '03', '04', '05', '06', '07', '09', '10'] 
        self.seq_ids['validation'] = ['00', '02', '03', '04', '05', '06', '07', '09', '10']
        self.seq_ids['test'] = ['08', '18']
        self.files = self.datapath_list(self.split)
        self.batch_size = self.params.batch_size

        self.frames = []
        for seq in self.seq_ids[self.split]:
            ply_path = join(self.path, self.split, "sequences", seq)
            frames = np.sort([vf[:-4] for vf in listdir(ply_path) if vf.endswith('.ply')])
            self.frames.append(frames)
        
        self.class_proportions = None
        self.load_proportions()
        return

    def read_ply(self, file_name):
        data = ost.read_ply(file_name)
        cloud_x = data['x']
        cloud_y = data['y']
        cloud_z = data['z']
        labels = (data['c']).astype(np.int32)
        labels = labels.reshape(len(labels), 1)
        return(np.c_[cloud_x,cloud_y,cloud_z],labels)

    
    
    
    def datapath_list(self, split):
        self.points_datapath = []
        # self.labels_datapath = []

        for seq in self.seq_ids[split]:
            point_seq_path = os.path.join(self.path, split, 'sequences', seq)
            point_seq_ply = os.listdir(point_seq_path)
            point_seq_ply.sort()
            self.points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_ply ]
        return list(np.random.permutation(self.points_datapath))
            
    def load_proportions(self):
        self.class_proportions = np.zeros((self.num_classes,), dtype=np.int32)
        for s_ind, (seq, seq_frames) in enumerate(zip(self.seq_ids[self.split], self.frames)):
            # Initiate dict
            # Check if inputs have already been computed
            frame_mode = 'single'
            seq_stat_file = join(self.path, "seq_stat", seq, 'stats_{:s}.pkl'.format(frame_mode))
        
            if exists(seq_stat_file):
                # Read pkl
                with open(seq_stat_file, 'rb') as f:
                    seq_class_frames, seq_proportions = pickle.load(f)
        
            else:
        
                print('Preparing seq {:s} class frames. (Long but one time only)'.format(seq))
            
                # Class frames as a boolean mask
                seq_class_frames = np.zeros((len(seq_frames), self.num_classes), dtype=np.bool)
            
                # Proportion of each class
                seq_proportions = np.zeros((self.num_classes,), dtype=np.int32)
            
                # Sequence path
                seq_path = join(self.path, self.split, "sequences", seq)
            
                # Read all frames
                for f_ind, frame_name in enumerate(seq_frames):
            
                    # Path of points and labels
                    label_file = join(seq_path, frame_name + '.ply')
            
                    # Read labels
                    _, input_labels = self.read_ply(label_file)
                    input_labels = self.learning_map[input_labels]
                    # Get present labels and there frequency
                    unique, counts = np.unique(input_labels, return_counts=True)
                    # Add this frame to the frame lists of all class present
                    frame_labels = np.array([self.label_to_idx[l] for l in unique], dtype=np.int32)
                    seq_class_frames[f_ind, frame_labels] = True
            
                    # Add proportions
                    seq_proportions[frame_labels] += counts
                                # Save pickle
                with open(seq_stat_file, 'wb') as f:
                    pickle.dump([seq_class_frames, seq_proportions], f)
        self.class_proportions += seq_proportions      
        return
    
   
    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.files)

    def get_input_labels(self, idx_list):
        points, labels = self.load_subsampled_clouds(idx_list)
        return points, labels


    def __getitem__(self, idx_list):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        ###################
        # Gather batch data
        ###################
        tp_list = []
        tl_list = []
        for p_i in idx_list:
            # read ply with data
            points, labels = self.load_subsampled_clouds(p_i)
            # Get points and labels
            points = points.astype(np.float32)
            # Stack batch
            tp_list += [points]
            tl_list += [labels]

        ###################
        # Concatenate batch
        ###################

        #show_ModelNet_examples(tp_list, cloud_normals=tn_list)
        stacked_points = np.concatenate(tp_list, axis=0)
        labels = np.array(tl_list, dtype=np.int32)
        labels = labels.squeeze()

        # Get the whole input list
        input_list = np.c_[stacked_points, labels]
        return input_list
    
    def load_subsampled_clouds(self,idx):
        # read ply with data
        input_points, input_labels = self.read_ply(self.files[idx])
        input_points = input_points[:8192,:]
        input_labels = input_labels[:8192]
        input_labels = self.learning_map[input_labels]
        return input_points, input_labels
    
    
    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """
        # Get original points
        points, _ = self.read_ply(file_path)
        return points


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class SemanticKittiAggSampler(Sampler):
    """Sampler for SemanticKittiAgg"""

    def __init__(self, dataset: AggDataset, params):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset
        self.params = params
        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = params.batch_limit
        return
    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return len(self.dataset.files)
    def __iter__(self):
        """
        Yield next batch indices here
        """

        gen_indices = np.random.permutation(len(self.dataset.files))
        input_points, _  = self.dataset.load_subsampled_clouds(0)
        n = input_points.shape[0]
        ################
        # Generator loop
        ################

        # Initialize concatenation lists
        ti_list = []
        batch_n = 0
        
        # Generator loop
        for p_i in gen_indices:
            # Size of picked cloud
            # In case batch is full, yield it and reset it
            if batch_n + n > self.batch_limit and batch_n > 0:
                yield np.array(ti_list, dtype=np.int32)
                ti_list = []
                batch_n = 0
            ti_list += [p_i]
            batch_n += n

        yield np.array(ti_list, dtype=np.int32)
        return 0



class SemanticKittiAggCustomBatch:
    """Custom batch definition with memory pinning for SemanticKittiAgg"""

    def __init__(self, input_list):
        input_array = np.array(input_list)        
        self.points = torch.from_numpy(input_array[:,:,0:3])
        self.labels = torch.from_numpy(input_array[:,:,3])
        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """
        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.labels = self.labels.pin_memory()
        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.labels = self.labels.to(device)
        return self



def SemanticKittiAggCollate(batch_data):
    return SemanticKittiAggCustomBatch(batch_data)


