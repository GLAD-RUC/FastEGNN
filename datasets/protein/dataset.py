import os
import math
import random
import numpy as np
import pickle as pkl

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

import MDAnalysis
import MDAnalysisData
from MDAnalysis import transformations
from MDAnalysis.analysis import distances

from utils.rotate import random_rotate


class MDAnalysisDataset(Dataset):
    """
    MDAnalysisDataset
    """
    def __init__(self, dataset_name, data_dir, virtual_channels, partition='train', max_samples=1e8, delta_frame=1, train_valid_test_ratio=None,
                 test_rot=False, test_trans=False, load_cached=False, cutoff_rate=0.0, backbone=False):
        super(MDAnalysisDataset, self).__init__()
        self.delta_frame = delta_frame
        self.partition = partition
        self.test_rot = test_rot
        self.test_trans = test_trans
        self.cutoff_rate = cutoff_rate
        self.backbone = backbone
        self.data_dir = data_dir
        self.virtual_channels = virtual_channels

        # Fast EGNN -- virtual node
        file_name = f'c_{self.cutoff_rate :.2f}_dt_{self.delta_frame}_{self.partition}.pkl'
        if backbone:
            file_name = os.path.join(data_dir, 'adk_backbone_processed', file_name)
        else:
            file_name = os.path.join(data_dir, 'adk_processed', file_name)

        load_cached = False
        if load_cached and os.path.exists(file_name):
            print(f'Loading processed data to {partition} dataset...')
            with open(file_name, 'rb') as f:
                self.data = pkl.load(f)
        else:
            print('Processing data ...')
            train_valid_test_cnt =  [0, 2481, 2481 + 827, 2481 + 827 + 863]
            if partition == 'train':
                self.data = self.process(train_valid_test_cnt[0], train_valid_test_cnt[1])
            elif partition == 'valid':
                self.data = self.process(train_valid_test_cnt[1], train_valid_test_cnt[2])
            elif partition == 'test':
                self.data = self.process(train_valid_test_cnt[2], train_valid_test_cnt[3])
            else:
                assert False

        self.num_node_r = self.data[0]['node_attr'].size(0)
        print(f'{partition} dataset total len: {len(self.data)}')
        print(self.data[0])
    

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, i):
        return self.data[i]
    

    def get_data(self):
        return self.data
        

    def process(self, l, r):
        adk = MDAnalysisData.datasets.fetch_adk_equilibrium(data_home=self.data_dir)
        adk_data = MDAnalysis.Universe(adk.topology, adk.trajectory)
        return Parallel(n_jobs=10)(delayed(self.get_graph_step)(i, adk_data) for i in tqdm(range(l, r)))
        

    def get_graph_step(self, t, adk_data):
        if self.backbone:
            ag = adk_data.select_atoms('backbone')
        else:
            ag = adk_data.atoms

        charges = torch.tensor(adk_data.atoms[ag.ix].charges).float().unsqueeze(-1)

        frame_0, frame_t = t, t + self.delta_frame

        ts_0, ts_t = None, None
        # Initial frame
        retry_0 = 0
        while retry_0 < 10:
            try:
                ts_0 = adk_data.trajectory[frame_0].copy()
                if not ts_0.has_velocities:
                    ts_0.velocities = adk_data.trajectory[frame_0 + 1].positions - adk_data.trajectory[frame_0].positions
                retry_0 = 11
            except OSError:
                print(f'Reading error at {frame_0}')
                retry_0 += 1
        assert retry_0 != 10, OSError(f'Falied to read positions by 10 times')

        # Final frame
        retry_t = 0
        while retry_t < 10:
            try:
                ts_t = adk_data.trajectory[frame_t].copy()
                if not ts_t.has_velocities:
                    ts_t.velocities = adk_data.trajectory[frame_t + 1].positions - adk_data.trajectory[frame_t].positions
                retry_t = 11
            except OSError:
                print(f'Reading error at {frame_t} t')
                retry_t += 1
        assert retry_t != 10, OSError(f'Falied to read velocity by 10 times')


        loc_0 = torch.tensor(ts_0.positions[ag.ix])
        vel_0 = torch.tensor(ts_0.velocities[ag.ix])

        loc_t = torch.tensor(ts_t.positions[ag.ix])
        vel_t = torch.tensor(ts_t.velocities[ag.ix])

        if self.test_rot and self.partition == 'test':
            rotate_matrix = random_rotate()
            rotate_matrix = rotate_matrix.to(loc_0.device).to(loc_0.dtype)
            loc_0 = loc_0 @ rotate_matrix
            loc_t = loc_t @ rotate_matrix
            vel_0 = vel_0 @ rotate_matrix

        if self.test_trans and self.partition == 'test':
            trans = np.random.randn(3) * ts_0.dimensions[:3] / 2
            trans = torch.from_numpy(trans).unsqueeze(0).to(loc_0.device).to(loc_0.dtype)
            loc_0 = loc_0 + trans
            loc_t = loc_t + trans


        # Edges
        edge_index = coo_matrix(distances.contact_matrix(loc_0.detach().numpy(), cutoff=10, returntype="sparse"))
        edge_index.setdiag(False)
        edge_index.eliminate_zeros()
        edge_index = torch.stack([torch.tensor(edge_index.row, dtype=torch.long),
                                torch.tensor(edge_index.col, dtype=torch.long)], dim=0)
        
        # Cutoff edges
        edge_index = self.cutoff_edge(edge_index, loc_0)

        # Edge attr
        edge_attr = torch.norm(loc_0[edge_index[0], :] - loc_0[edge_index[1], :], p=2, dim=1).unsqueeze(-1)

        # Node Feat
        feat_node_velocity = torch.sqrt(torch.sum(vel_0 ** 2, dim=1)).unsqueeze(1)
        feat_node_charge = charges
        node_feat = torch.cat([feat_node_velocity, feat_node_charge / feat_node_charge.max()], dim=1)

        # Virtual node loc = mean
        loc_mean = torch.mean(loc_0, dim=0).unsqueeze(-1).repeat(1, self.virtual_channels).unsqueeze(0)  # [1, 3, C]

        # For VN-EGNN
        if self.virtual_channels > 1:
            virtual_fibonacci = self.get_virtual_fibonacci_pos(loc_0).unsqueeze(0)
        else:
            virtual_fibonacci = None

        return Data(edge_index=edge_index, edge_attr=edge_attr, loc_0=loc_0, loc_t=loc_t, vel_0=vel_0, \
                    node_feat=node_feat, node_attr=charges, loc_mean=loc_mean, virtual_fibonacci=virtual_fibonacci)

    
    def get_virtual_fibonacci_pos(self, node_loc):  # node_loc: [N, 3]
        center = torch.mean(node_loc, dim=0).unsqueeze(0)  # [1, 3]
        dist = node_loc - center  # [N, 3]
        dist = torch.norm(dist, dim=1)  # [N]
        radius = torch.max(dist)  # scalar

        # From: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/26127012#26127012
        # For: VN-EGNN
        def fibonacci_sphere(samples=1000):
            points = []
            phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

            for i in range(samples):
                y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
                radius = math.sqrt(1 - y * y)  # radius at y

                theta = phi * i  # golden angle increment

                x = math.cos(theta) * radius
                z = math.sin(theta) * radius

                points.append([x, y, z])

            return points

        points = fibonacci_sphere(self.virtual_channels)
        points = torch.tensor(points)
        points = points * radius
        points = points + center
        return points


    def cutoff_edge(self, edge_index, loc_0):
        edge_dist = torch.norm(loc_0[edge_index[0]] - loc_0[edge_index[1]], p=2, dim=1)
        _, id_chosen = torch.sort(edge_dist)
        id_chosen = id_chosen[:int(id_chosen.size(0) * (1 - self.cutoff_rate))]
        edge_index = edge_index[:, id_chosen]
        return edge_index
    

    @staticmethod
    # For basic model EGHN
    def get_local_edge(data_dir):
        adk = MDAnalysisData.datasets.fetch_adk_equilibrium(data_home=data_dir)
        adk_data = MDAnalysis.Universe(adk.topology, adk.trajectory)
        local_edge_index = torch.stack([torch.tensor(adk_data.bonds.indices[:, 0], dtype=torch.long),
                                        torch.tensor(adk_data.bonds.indices[:, 1], dtype=torch.long)], dim=0)
        local_edge_attr = torch.tensor([bond.length() for bond in adk_data.bonds]).unsqueeze(-1).float()
        return local_edge_index, local_edge_attr
        