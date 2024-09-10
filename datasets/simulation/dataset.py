import os
import h5py
import random
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph

from utils.rotate import random_rotate_y


class Simulation(Dataset):
    def __init__(self, dataset_name, data_dir, virtual_channels, partition='train', max_samples=1e8, delta_t=15, cutoff_rate=0., device='cpu'):
        super(Simulation, self).__init__()
        self.partition = partition
        self.data_dir = data_dir
        self.cutoff_rate = cutoff_rate
        self.virtual_channels = virtual_channels
        self.delta_t = delta_t
        self.max_samples = max_samples

        file_path = os.path.join(data_dir, dataset_name, f'{partition}.h5')
        print(f'file_path: {file_path}')

        print('Processing data ...')
        self.data = self.process(file_path, device)  # Process in GPU
        random.shuffle(self.data)
        
        print(f'{partition} dataset total len: {len(self.data)}')
        print(self.data[0])

    
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, i):
        return self.data[i]


    def process(self, path, device):
        data = []
        with h5py.File(path, 'r') as file:
            keys = list(file.keys())
            for k in tqdm(keys, desc='Graph'):
                particle_type = file[k]['particle_type']
                particle_type = torch.tensor(np.array(particle_type)).float().unsqueeze(-1)
                postion = file[k]['position']
                postion = torch.tensor(np.array(postion)).float()
                # print(postion.size(0), len(data))
                frames = postion.size(0)
                frames = [random.randint(0, 250) for _ in range(min(15, self.max_samples - len(data)))]  # Select 15 frames from former 150 frames
                # frames.sort()
                for frame in frames:
                    vel_frame = postion[frame + 1, :, :] - postion[frame, :, :]
                    data.append(self.get_graph_step(postion[frame, :, :].to(device),  # loc_0
                                                    vel_frame.to(device),  # vel_0
                                                    postion[frame + self.delta_t, :, :].to(device),  # loc_t
                                                    particle_type.to(device)).to('cpu'))  # node_type)
                if len(data) >= self.max_samples:
                    break
        return data

    
    def get_graph_step(self, loc_0, vel_0, loc_t, node_type):
        roteta_matrix = random_rotate_y()
        roteta_matrix = roteta_matrix.to(loc_0.device).to(torch.float)

        if self.partition == 'test':
            loc_0 = loc_0 @ roteta_matrix
            loc_t = loc_t @ roteta_matrix
            vel_0 = vel_0 @ roteta_matrix

        # Edge
        edge_index = radius_graph(loc_0, r=0.035, max_num_neighbors=100000)
        edge_index = self.cutoff_edge(edge_index, loc_0)
        edge_attr = torch.norm(loc_0[edge_index[0], :] - loc_0[edge_index[1], :], p=2, dim=1).unsqueeze(-1)

        # Node Feat
        feat_node_velocity = torch.sqrt(torch.sum(vel_0 ** 2, dim=1)).unsqueeze(1)
        feat_node_charge = node_type
        node_feat = torch.cat([feat_node_velocity, feat_node_charge / feat_node_charge.max()], dim=1)

        # Virtual node loc = mean
        loc_mean = torch.mean(loc_0, dim=0).unsqueeze(-1).repeat(1, self.virtual_channels).unsqueeze(0)  # [1, 3, C]

        return Data(edge_index=edge_index, edge_attr=edge_attr, loc_0=loc_0, loc_t=loc_t, vel_0=vel_0, \
                    node_feat=node_feat, node_attr=node_type, loc_mean=loc_mean)


    def cutoff_edge(self, edge_index, loc_0):
        edge_dist = torch.norm(loc_0[edge_index[0]] - loc_0[edge_index[1]], p=2, dim=1)
        _, id_chosen = torch.sort(edge_dist)
        id_chosen = id_chosen[:int(id_chosen.size(0) * (1 - self.cutoff_rate))]
        edge_index = edge_index[:, id_chosen]
        return edge_index
    