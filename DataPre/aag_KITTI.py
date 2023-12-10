import numpy as np
import OSToolBox as ost
from sklearn.neighbors import KDTree


def cube_points(pc, center_point, dia):
    points = []
    cx,cy,cz = center_point[:3] # coordinates of a point which you try to examine
    dX = dY = dZ = dia
    for i in range(pc.shape[0]):
        px,py,pz = pc[i,:3] # coordinates of a point which you try to examine
        if abs(cx-px) < dX:
            if abs(cy-py) < dY:
                if abs(cz-pz) < dZ:
                    points.append(pc[i,:])
    return(points)

def aag(root, pc, knn, dia, sample_knn, seq_id, block):
    lo = [0]*sample_knn
    lo_ind = 0
    while len(lo) >= sample_knn:
        if lo_ind != 0:
            pc = lo_arr
        n_random = int(((pc.shape[0])/knn)*2) #number of seed points
        ind_c = np.random.choice(pc.shape[0], n_random, replace=False) #ind of randomly selected seed points
        x_random = pc[ind_c,:] #randomly selected seed points
        #selecting the sampling area for the selected seed points
        tree = KDTree(pc[:,:3])
        distances, indices = tree.query(x_random[:,:3], sample_knn)
        zeros = np.zeros((pc.shape[0],1))
        pc = np.append(pc, zeros, axis=1)
        centroids = []
        for i in range(x_random.shape[0]):
            centroids.append(pc[indices[i],:4])
            pc[indices[i],4] = 1
        lo = []    
        for i in range(pc.shape[0]):
            if pc[i,4] == 0:
                lo.append(pc[i,:4])
        if len(lo) > 0:
            lo_arr = np.stack(lo, axis=0)
        del n_random, ind_c, tree, pc
        for i in range(x_random.shape[0]):
            c = x_random[i,:]
            points = cube_points(centroids[i], c, dia)
            while len(points) < knn:
                dia += 0.1
                points = cube_points(centroids[i], c, dia)
                # print("Growing...")
            box_points = np.random.permutation(points)[:knn,:]
            filename = root + "/aag/" + seq_id + "/" + str(block) + str(i) + str(lo_ind) + ".ply"
            # ost.ply_write(filename, box_points, ['x','y','z','c'])
            dia = 0.5
        lo_ind += 1




        
        
        
        
