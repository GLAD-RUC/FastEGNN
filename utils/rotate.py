import torch
import random
import numpy as np

def rotx(theta):
    """
    以X轴为旋转轴的旋转矩阵
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cos_theta, -sin_theta],
                                [0, sin_theta, cos_theta]])
    return rotation_matrix

def roty(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, 0, sin_theta],
                                [0,         1,         0],
                                [-sin_theta, 0, cos_theta]])
    return rotation_matrix

def rotz(theta):
    """
    以Z轴为旋转轴的旋转矩阵
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                [sin_theta, cos_theta, 0],
                                [0, 0, 1]])
    return rotation_matrix

def random_rotate():
    x, y, z = random.randint(0, 360), random.randint(0, 360), random.randint(0, 360)
    x, y, z = np.radians(x), np.radians(y), np.radians(z) 
    rotate_matrix = rotx(x)
    rotate_matrix = rotate_matrix @ roty(y)
    rotate_matrix = rotate_matrix @ rotz(z)
    rotate_matrix = torch.from_numpy(rotate_matrix)
    return rotate_matrix

def random_rotate_y():
    y = random.randint(0, 360)
    y = np.radians(y)
    rotate_matrix = rotx(y)
    rotate_matrix = torch.from_numpy(rotate_matrix)
    return rotate_matrix

