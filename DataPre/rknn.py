import numpy as np
from sklearn.neighbors import KDTree
import OSToolBox as ost


def random_knn(root, pc, knn, seq, block):
    n_random = int(((pc.shape[0])/knn)*2) #number of seed points
    ind_c = np.random.choice(pc.shape[0], n_random, replace=False) #ind of randomly selected seed points
    c_random = pc[ind_c,:3] #randomly selected seed points

    tree = KDTree(pc[:,:3], leaf_size=10)
    _, pnt_inds = tree.query(c_random, knn)
    for i in range(c_random.shape[0]):
        points = pc[pnt_inds[i],:]
        points = points[np.random.permutation(points.shape[0]), :]
        filename = root + "/rknn/sequences/" + seq + "/" + str(block) + str(i) + ".ply"
        ost.ply_write(filename, points, ['x','y','z','c'])

