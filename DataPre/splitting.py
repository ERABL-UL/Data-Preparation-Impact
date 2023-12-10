import numpy as np
import OSToolBox as ost
import gc

def read_ply(filename):
    data0 = ost.read_ply(filename)
    cloud_x = data0['x']
    cloud_y = data0['y']
    cloud_z = data0['z']
    labels = data0['semantic']
    return np.c_[cloud_x, cloud_y, cloud_z, labels]

def split_to_blocks(filename, block_size, overlap, knn, min_point):
    pc = read_ply(filename)
    pc = pc[np.newaxis,:,:]
    deltaX = pc[0, :, 0].max() - pc[0, :, 0].min()
    deltaY = pc[0, :, 1].max() - pc[0, :, 1].min()
    deltaZ = pc[0, :, 2].max() - pc[0, :, 2].min()
    first_block = np.array([[pc[0, :, 0].min(), pc[0, :, 1].min(), pc[0, :, 2].min()]])
    #Generating X, Y, Z for the next blocks
    x = np.zeros((int(deltaX/block_size)+1),)
    for i in range(int(deltaX/block_size)+1):
        x[i] = first_block[0,0]+i*(block_size)
    
    y = np.zeros((int(deltaY/(block_size-overlap))+1),)
    for j in range(int(deltaY/(block_size-overlap))+1):
        y[j] = first_block[0,1]+j*(block_size-overlap)
    
    z = first_block[0,2]
    
    block_coordinates = np.zeros((x.shape[0]*y.shape[0]*z.size,3))
    
    for i in range(0,x.shape[0]):
        block_coordinates[i*y.shape[0]*z.size:y.shape[0]*z.size+i*y.shape[0]*z.size,0] = x[i]
    for n in range(int(block_coordinates.shape[0]/y.shape[0])):
        for j in range(0,y.shape[0]):   
            block_coordinates[j*z.size+n*z.size*y.shape[0]:(j+1)*z.size+n*z.size*y.shape[0],1] = y[j]
    for k in range(x.shape[0]*y.shape[0]):
        block_coordinates[:,2] = z
    block_coordinates = block_coordinates[np.newaxis ,:, :]
    #Feeding the point into the blocks           
    point_cloud_blocks = [0] * block_coordinates.shape[1]
    pc_blocks = []
    for i in range(int(block_coordinates.shape[1])):
        point_cloud_blocks[i] = pc[:, (pc[0, :, 0] > block_coordinates[0, i, 0]) &
                                    (pc[0, :, 0] <= block_coordinates[0, i, 0]+block_size) &
                                    (pc[0, :, 1] > block_coordinates[0, i, 1]) &
                                    (pc[0, :, 1] <= block_coordinates[0, i, 1]+block_size)]
                                    # (pc[0, :, 2] > block_coordinates[0, i, 2]) &
                                    # (pc[0, :, 2] <= block_coordinates[0, i, 2]+deltaZ)]
        # print((i+1)/int(block_coordinates.shape[1])*100)
        if point_cloud_blocks[i].shape[1] > min_point: pc_blocks.append(point_cloud_blocks[i][0,:,:])
        point_cloud_blocks[i] = []
    del pc, deltaX, deltaY, deltaZ, first_block, x, y, z, block_coordinates, point_cloud_blocks
    gc.collect()
    return pc_blocks
