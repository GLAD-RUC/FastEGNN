import pickle as pkl
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from utils.rotate import random_rotate


class NBodySystemDataset(Dataset):
    """
    NBodySystemDataset
    """
    def __init__(self, dataset_name, data_dir, virtual_channels, partition='train', max_samples=1e8, frame_0=30, frame_T=40, cutoff_rate=0., device='cpu'):
        super(NBodySystemDataset, self).__init__()
        self.partition = partition
        self.data_dir = data_dir
        self.frame_0, self.frame_T = frame_0, frame_T
        self.cutoff_rate = cutoff_rate
        self.virtual_channels = virtual_channels

        self.suffix = f'{self.partition}_charged{dataset_name}'

        self.max_samples = int(max_samples)
        loc, vel, _, charges, _ = self.load_data()

        print('Processing data ...')
        self.data = self.process(loc, vel, charges, device)  # Process in GPU
        print(f'{partition} dataset total len: {len(self.data)}')
        print(self.data[0])


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, i):
        return self.data[i]
    

    def load_data(self):
        loc = np.load(f'{self.data_dir}/loc_{self.suffix}.npy')
        vel = np.load(f'{self.data_dir}/vel_{self.suffix}.npy')
        charges = np.load(f'{self.data_dir}/charges_{self.suffix}.npy')
        edges = np.load(f'{self.data_dir}/edges_{self.suffix}.npy')
        with open(f'{self.data_dir}/cfg_{self.suffix}.pkl', 'rb') as f:
            cfg = pkl.load(f)

        self.num_node_r = loc.shape[-2]
        return loc, vel, edges, charges, cfg
    
    
    def process(self, loc, vel, charges, device):
        charges = torch.Tensor(charges)
        loc, vel = torch.Tensor(loc), torch.Tensor(vel)  

        loc = loc[0:self.max_samples, :, :, :]           
        vel = vel[0:self.max_samples, :, :, :]          
        charges = charges[0: self.max_samples]  # [num_systems, num_node_r, 1]

        loc_0, loc_t = loc[:, self.frame_0, :, :], loc[:, self.frame_T, :, :]  # [num_systems, num_node_r, 3]
        vel_0, vel_t = vel[:, self.frame_0, :, :], vel[:, self.frame_T, :, :]  # [num_systems, num_node_r, 3]

        num_systems, num_node_r, _ = charges.size()
        loc_0, loc_t, vel_0, charges = loc_0.to(device), loc_t.to(device), vel_0.to(device), charges.to(device)

        data = []
        for i in tqdm(range(num_systems)):
            data.append(self.get_graph_step(loc_0[i, :, :], vel_0[i, :, :], charges[i, :, :], loc_t[i, :, :]).to('cpu'))

        return data
    

    def get_graph_step(self, loc_0, vel_0, charges, loc_t):
        rotate_matrix = random_rotate()
        rotate_matrix = rotate_matrix.to(loc_0.device).to(torch.float)

        if self.partition == 'test':
            loc_0 = loc_0 @ rotate_matrix
            loc_t = loc_t @ rotate_matrix
            vel_0 = vel_0 @ rotate_matrix

        # Edge
        edge_index = self.cutoff_edge(loc_0)
        edge_attr = torch.norm(loc_0[edge_index[0], :] - loc_0[edge_index[1], :], p=2, dim=1).unsqueeze(-1)

        # Node Feat
        feat_node_velocity = torch.sqrt(torch.sum(vel_0 ** 2, dim=1)).unsqueeze(1)
        feat_node_charge = charges
        node_feat = torch.cat([feat_node_velocity, feat_node_charge / feat_node_charge.max()], dim=1)

        # Virtual node loc = mean
        loc_mean = torch.mean(loc_0, dim=0).unsqueeze(-1).repeat(1, self.virtual_channels).unsqueeze(0)  # [1, 3, C]

        return Data(edge_index=edge_index, edge_attr=edge_attr, loc_0=loc_0, loc_t=loc_t, vel_0=vel_0, \
                    node_feat=node_feat, node_attr=charges, loc_mean=loc_mean)


    def cutoff_edge(self, loc_0):
        # Complete Graph and Cutoff    
        num_node_r = loc_0.size(0)
        dist = torch.cdist(loc_0, loc_0, p=2)  # [num_node_r, num_node_r]
        dist += torch.eye(num_node_r).to(loc_0.device) * 1e18  # [num_node_r, num_node_r]
        num_edge_rr_chosen = int(num_node_r * (num_node_r - 1) * (1 - self.cutoff_rate))
        _, id_chosen = torch.topk(dist.view(num_node_r * num_node_r), num_edge_rr_chosen, dim=0, largest=False)
        edge_rr = torch.cat([
            id_chosen.div(num_node_r, rounding_mode='trunc').unsqueeze(0), 
            id_chosen.remainder(num_node_r ).unsqueeze(0)
        ], dim=0).long()  # [2, num_edge_rr_chosen]
        return edge_rr
    
