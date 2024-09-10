import random
import torch
import numpy as np

from torch import nn


from utils.rotate import random_rotate
from models.FastEGNN import FastEGNN

# Declare FastEGNN
model = FastEGNN(node_feat_nf=1, node_attr_nf=0, edge_attr_nf=1, hidden_nf=64, virtual_channels=3, 
                 device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, 
                 normalize=False, tanh=False, gravity=None)


# Random generate data
node_cnt = 10
edge_cnt = 20

node = [i for i in range(node_cnt)]
data_batch = [0 for i in range(node_cnt)]
coordinates = [[random.uniform(0, 10), random.uniform(0, 10), random.uniform(0, 10)] for _ in range(node_cnt)]
velocities = [[random.uniform(0, 10), random.uniform(0, 10), random.uniform(0, 10)] for _ in range(node_cnt)]
node_feat = [[random.uniform(0, 10)] for _ in range(node_cnt)]
edges = [[int(random.uniform(0, 10)) for _ in range(edge_cnt)], 
         [int(random.uniform(0, 10)) for _ in range(edge_cnt)]]
edge_attr = [[random.uniform(0, 10)] for _ in range(edge_cnt)]

node = torch.tensor(node)
data_batch = torch.tensor(data_batch, dtype=torch.long)
coordinates = torch.tensor(coordinates, dtype=torch.float)
velocities = torch.tensor(velocities, dtype=torch.float)
node_feat = torch.tensor(node_feat, dtype=torch.float)
edges = torch.tensor(edges, dtype=torch.long)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

rotate_matrix = random_rotate().to(torch.float)
translation = torch.randn(size=(3, )) * 5
coordinates_r = coordinates @ rotate_matrix + translation
velocities_r = velocities @ rotate_matrix


# Result Before Rotate
loc_mean = torch.mean(coordinates, dim=0).unsqueeze(-1).repeat(1, 3).unsqueeze(0)
result_before_rotate, _ = model(node_loc=coordinates.detach(), node_vel=velocities.detach(), 
                          node_attr=None, node_feat=node_feat.detach(), edge_index=edges, 
                          loc_mean=loc_mean.detach(), data_batch=data_batch, edge_attr=edge_attr)


loc_mean_r = torch.mean(coordinates_r, dim=0).unsqueeze(-1).repeat(1, 3).unsqueeze(0)
result_after_rotate, _ = model(node_loc=coordinates_r.detach(), node_vel=velocities_r.detach(), 
                          node_attr=None, node_feat=node_feat, edge_index=edges, 
                          loc_mean=loc_mean_r.detach(), data_batch=data_batch, edge_attr=edge_attr)

print(f'result_before_rotate: {result_before_rotate}')
print(f'result_before_rotate @ rotate_matrix + translation: {result_before_rotate @ rotate_matrix + translation}')
print(f'result_after_rotate: {result_after_rotate}')


# Test
assert torch.allclose(result_before_rotate @ rotate_matrix + translation, result_after_rotate, atol=1e-4)
print("Model is SE(3) Equivariant")



# Output:
"""
result_before_rotate: tensor([[ 3.4415,  0.3903,  4.3827],
        [ 6.2887,  9.0439,  9.3648],
        [ 0.4035,  2.1624,  7.8279],
        [ 9.1274,  8.0141,  1.9432],
        [ 0.2296,  7.9948,  6.4317],
        [ 9.6690,  2.2172,  8.6671],
        [ 0.3381,  7.4376,  4.1592],
        [ 8.5897,  5.6957,  7.8782],
        [ 8.8693,  5.1400,  6.7728],
        [ 6.2256,  3.5971, -0.4030]], grad_fn=<AddBackward0>)
result_before_rotate @ rotate_matrix + translation: tensor([[ -6.4509,   8.9788,  -9.2361],
        [-10.3862,   8.8141,   0.3711],
        [ -5.5811,  12.8571,  -6.3309],
        [ -8.8254,   1.7369,  -3.0458],
        [ -3.9086,  10.2095,  -1.2133],
        [-13.7023,   8.3216,  -6.5048],
        [ -2.8328,   8.5360,  -2.4495],
        [-11.9210,   7.3214,  -3.3881],
        [-11.6229,   6.4616,  -4.2716],
        [ -5.7247,   2.7832,  -7.8130]], grad_fn=<AddBackward0>)
result_after_rotate: tensor([[ -6.4509,   8.9788,  -9.2361],
        [-10.3862,   8.8141,   0.3711],
        [ -5.5811,  12.8571,  -6.3309],
        [ -8.8254,   1.7369,  -3.0458],
        [ -3.9086,  10.2095,  -1.2133],
        [-13.7023,   8.3216,  -6.5048],
        [ -2.8328,   8.5360,  -2.4495],
        [-11.9210,   7.3214,  -3.3881],
        [-11.6229,   6.4616,  -4.2716],
        [ -5.7247,   2.7832,  -7.8130]], grad_fn=<AddBackward0>)
Model is SE(3) Equivariant
"""