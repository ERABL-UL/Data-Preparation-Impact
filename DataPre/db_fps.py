import numpy as np
import torch
from sklearn.neighbors import KDTree
import multiprocessing as mp
from kde_func import dens_kde
from dgl.geometry import farthest_point_sampler as FPS
import OSToolBox as ost

def density_based_fps(pc, root, downsampling_rate, high_density_rate, low_density_rate, knn, b, seq_id):
    lo_len = knn
    lo_ind = 0
    while lo_len >= knn:
        if lo_ind != 0:
            pc = lo
            ds_pc = pc[::downsampling_rate]
        else:
            ds_pc = pc[::downsampling_rate]
        cores = mp.cpu_count()
        pool = mp.Pool(processes = cores)
        torun = np.array_split(ds_pc[:,:3].T, cores, axis=1)
        density = np.hstack(pool.map(dens_kde, torun))
        pool.terminate()
        pool.join() 
        density_norm = density/(density.max()/100)
        density_norm = density_norm[:, np.newaxis]
        ds_pc = np.c_[ds_pc, density_norm]
        #Calculating the number of centroids for each denisty-wise zones in each block
        n_centroids = int(pc.shape[0]/knn)+1
        if n_centroids>2:
            n_centroids_high = n_centroids_low = int(n_centroids/2)+1
            n_centroids_med = int(n_centroids)
        else:
            n_centroids_high = n_centroids_low = n_centroids_med = 1

        #Getting the farthest points in the three zones as the seed points
        ds_pc_fsp_high =[point for point, density in zip(ds_pc, density_norm) if density > high_density_rate]
        if len(ds_pc_fsp_high) > 0:
            ds_pc_fsp_high =  np.vstack(ds_pc_fsp_high)
            farthest_pts_high = ds_pc_fsp_high[FPS(torch.unsqueeze(torch.Tensor(ds_pc_fsp_high[:,:3]),0), n_centroids_high).numpy().T[:,0]]
        
        ds_pc_fsp_med = [point for point, density in zip(ds_pc, density_norm) if low_density_rate < density < high_density_rate]
        if len(ds_pc_fsp_med) > 0:
            ds_pc_fsp_med =  np.vstack(ds_pc_fsp_med)
            farthest_pts_med = ds_pc_fsp_med[FPS(torch.unsqueeze(torch.Tensor(ds_pc_fsp_med[:,:3]),0), n_centroids_med).numpy().T[:,0]]
        
        ds_pc_fsp_low = [point for point, density in zip(ds_pc, density_norm) if density < low_density_rate]
        if len(ds_pc_fsp_low) > 0:
            ds_pc_fsp_low =  np.vstack(ds_pc_fsp_low)
            farthest_pts_low = ds_pc_fsp_low[FPS(torch.unsqueeze(torch.Tensor(ds_pc_fsp_low[:,:3]),0), n_centroids_low).numpy().T[:,0]]
        
        if len(ds_pc_fsp_high) > 0 and len(ds_pc_fsp_low) > 0 and len(ds_pc_fsp_med) > 0:                  
            farthest_pts_array = np.vstack((farthest_pts_low, farthest_pts_med, farthest_pts_high))
        elif len(ds_pc_fsp_high) == 0 :
            farthest_pts_array = np.vstack((farthest_pts_low, farthest_pts_med))
        elif len(ds_pc_fsp_med) == 0 :
            farthest_pts_array = np.vstack((farthest_pts_low, farthest_pts_high))
        elif len(ds_pc_fsp_low) == 0 :
            farthest_pts_array = np.vstack((farthest_pts_med, farthest_pts_high))
            
        tree = KDTree(pc[:,:3])
        _, indices = tree.query(farthest_pts_array[:,:3], knn)
        #Saving the group of points
        zeros = np.zeros((pc.shape[0],1))
        pc = np.append(pc, zeros, axis=1)
        for i in range(farthest_pts_array.shape[0]):
            # save_path = root + '/db/sequences/' + seq_id + '/' + str(b) + str(i) + str(lo_ind) + '.ply'
            # ost.ply_write(save_path, pc[indices[i],:4], ['x','y','z','c'])
            pc[indices[i],4] = 1
        lo = pc[pc[:, 4] == 0, :4]
        lo_len = lo.shape[0]
        lo_ind += 1



        



