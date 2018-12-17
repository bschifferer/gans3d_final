import os
import scipy.io as io
import scipy.ndimage as nd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.font_manager import FontProperties

def loadVoxel(path, cube_len=64):
    """loadVoxel
    
    Function loads a single voxel file (.mat) and scales it to the correct pixel size
    
    Args:
        path (string): Path to the voxel file (including file name)
        cube_len (int): Pixel length of one side of the cube

    Returns:
        voxels (nparray): The loaded file as np array
    """
    voxels = io.loadmat(path)['instance']
    # Padds the voxel with 0 in each dimension
    voxels = np.pad(voxels,(1,1),'constant',constant_values=(0,0))
    # Scales the voxel to the requested cube sizue
    voxels = nd.zoom(voxels, (cube_len/voxels.shape[0],cube_len/voxels.shape[1],cube_len/voxels.shape[2]), mode='constant', order=0)
    return voxels

def loadObjects(PATH, cube_len=64, num_obj=None):
    """loadObjects
    
    Function loads all relevant .mat files (voxekls) in the PATH (list)) 
      
    Args:
        path (string or list): Path /paths in which all files are
        cube_len (int): Pixel length of one side of the cube
        num_obj: Number of objects per path

    Returns:
        volumeBatch (nparray): The loaded elements from the directory
    """
    # If PATH is a list, iterate over the list, if not load it as string
    if isinstance(PATH, (list,)):
        fileList = []
        for p in PATH:
            # List all .mat files
            fileList_tmp = [p + f for f in os.listdir(p) if f.endswith('.mat')]
            # If there is a limit per class, filter on the limit
            if num_obj != None:
                fileList_tmp = fileList_tmp[0:num_obj]
            fileList = fileList + fileList_tmp
    else:
        fileList = [PATH + f for f in os.listdir(PATH) if f.endswith('.mat')]
        if num_obj != None:
            fileList = fileList[0:num_obj]
    # Read all files from the list
    volumeBatch = np.asarray([loadVoxel(f, cube_len) for f in fileList], dtype=np.bool)
    return volumeBatch

def createPath(path):
    """createPath
    
    Function creates a path, if the path does not exist
    
    Args:
        path (string): Path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def visImage(obj, path, angles = [30, 200, 230, 360]):
    """visImage
    
    Function visualize the voxel and saves the visualization to disk
    
    Args:
        obj (nparray): 
        path (string): Path to create
    """
    fig = plt.figure()
    
    vol = obj
    
    ind = np.argwhere(obj > 0.5)
    
    xs = ind[:,0]
    ys = ind[:,1]
    zs = ind[:,2]
    
    i = 1
    for angle in angles:
        ax = fig.add_subplot(220+i, projection='3d')
        ax.scatter(xs, ys, zs)
        ax.view_init(angle)
        i += 1
        
    fig.savefig(path + '_' + '.png', dpi=200)
    plt.close(fig)

def visImage_jupyter(objs):
    fig = plt.figure()
    
    i = 1
    for obj in objs:
        vol = obj
        ind = np.argwhere(obj > 0.5)
        xs = ind[:,0]
        ys = ind[:,1]
        zs = ind[:,2]
        ax = fig.add_subplot(220 + i, projection='3d')
        ax.scatter(xs, ys, zs)
        ax.view_init(30)
        i += 1
        
    plt.show()   

def visInterpolation(objs, path):
    fig = plt.figure(figsize=(4*len(objs[0]), 4))
    
    i = 1
    for obj in objs[0]:
        vol = obj
        ind = np.argwhere(obj > 0.5)
        xs = ind[:,0]
        ys = ind[:,1]
        zs = ind[:,2]
        ax = fig.add_subplot(len(objs[0]) * 10 + 100 + i, projection='3d')
        ax.scatter(xs, ys, zs)
        ax.view_init(30)
        i += 1
        
    fig.savefig(path + '_' + '.png')
    plt.close(fig)
    
def visInterpolation_jupyter(objs):
    fig = plt.figure(figsize=(4*len(objs[0]), 4))
    
    i = 1
    for obj in objs[0]:
        vol = obj
        ind = np.argwhere(obj > 0.5)
        xs = ind[:,0]
        ys = ind[:,1]
        zs = ind[:,2]
        ax = fig.add_subplot(len(objs[0]) * 10 + 100 + i, projection='3d')
        ax.scatter(xs, ys, zs)
        ax.view_init(30)
        i += 1
    
    plt.show()
    
def plotLineGraph(data_dict, path):
    fontP = FontProperties()
    fontP.set_size('small')
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    for key in data_dict.keys():
        ax.plot(data_dict[key], label=key)
    plt.legend(prop=fontP)
    fig.savefig(path)   
    plt.close(fig)
    
def plotLineGraph_jupyter(data_dict):
    fontP = FontProperties()
    fontP.set_size('small')
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    for key in data_dict.keys():
        ax.plot(data_dict[key], label=key)
    plt.legend(prop=fontP)
    plt.show()  