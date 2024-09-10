from math import pi as PI
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential

from torch_geometric.nn import MessagePassing, SumAggregation, radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.typing import OptTensor

from torch_scatter import scatter_mean
from torch_geometric.nn import global_mean_pool

class SchNet_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer (with SchNet)
    """

    def __init__(self, node_feat_nf, node_feat_out_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels,
                 act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, gravity=None):
        super(SchNet_GCL, self).__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.hiddden_nf = hidden_nf
        self.node_feat_out_nf = node_feat_out_nf
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        # For Fast-EGNN
        self.virtual_channels = virtual_channels

        # SchNet Layer
        self.SchNetLayer = SchNet(hidden_channels=hidden_nf, max_num_neighbors=1000, cutoff=10, num_interactions=1)

        ## MLPS
        self.edge_mlp = nn.Sequential(  # \phi_{e}
            nn.Linear(2 * node_feat_nf + edge_coords_nf + edge_attr_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.edge_mlp_virtual = nn.Sequential(  # \phi_{ev}
            nn.Linear(2 * node_feat_nf + edge_coords_nf + virtual_channels, hidden_nf),  # No edge_feat
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )


        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )

            self.att_mlp_virtual = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )
            

        def get_coord_mlp():
            layer = nn.Linear(hidden_nf, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            coord_mlp = []
            coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
            coord_mlp.append(act_fn)
            coord_mlp.append(layer)
            if self.tanh:
                coord_mlp.append(nn.Tanh())
            
            return coord_mlp
        

        self.coord_mlp_r = nn.Sequential(*get_coord_mlp())  # \phi_{x}
        self.coord_mlp_r_virtual = nn.Sequential(*get_coord_mlp())  # \phi_{xv}
        self.coord_mlp_v_virtual = nn.Sequential(*get_coord_mlp())  # \phi_{X}

        # Velocity
        self.coord_mlp_vel = nn.Sequential(  # \phi_{v}
            nn.Linear(node_feat_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1)
        )

        # Gravity
        self.gravity = gravity
        if self.gravity is not None:
            self.gravity_mlp = nn.Sequential(
                nn.Linear(node_feat_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 1)
            )

        self.node_mlp = nn.Sequential(  # \phi_{h}
            nn.Linear(hidden_nf + hidden_nf + virtual_channels * hidden_nf + node_attr_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, node_feat_out_nf)
        )

        self.node_mlp_virtual = nn.Sequential(  # \phi_{hv}
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, node_feat_out_nf)
        )


    def edge_model(self, source, target, radial, edge_attr):
        out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out
    
    # [batch_node, H, 1]  [batch_node, H, C], [batch_node, 1, C], [batch_node, C, C] -> [batch_node, H, C]
    def edge_mode_virtual(self, feat_R, feat_V, radial, mix_V):
        feat_R = feat_R.repeat(1, 1, self.virtual_channels)  # [batch_size, H, C]

        out = torch.cat([feat_R, feat_V, radial, mix_V], dim=1)
        out = self.edge_mlp_virtual(out.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, C, H]
        if self.attention:
            att_val = self.att_mlp_virtual(out.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, C, 1]
            out = out * att_val  # [batch_node, C, H]
        return out


    def coord_model(self, node_feat, coord, node_vel, edge_index, coord_diff, edge_feat, virtual_edge_feat, virtual_coord_diff):
        # virtual_edge_feat: [batch_node, H, C], virtual_coord_diff: [batch_node, 3, C]
        trans_v = torch.mean(-virtual_coord_diff * self.coord_mlp_r_virtual(virtual_edge_feat.permute(0, 2, 1)).permute(0, 2, 1), dim=-1)  # [batch_node, 3]
        coord = coord + trans_v

        if self.gravity is not None:
            coord = coord + self.gravity_mlp(node_feat) * self.gravity  # Gravity

        return coord
    
    def coord_model_virtual(self, virtual_coord, virtual_edge_feat, virtual_coord_diff, data_batch):
        trans = virtual_coord_diff * self.coord_mlp_v_virtual(virtual_edge_feat.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, 3, C]
        agg = global_mean_pool(trans.reshape(trans.size(0), -1), data_batch).reshape(-1, 3, self.virtual_channels)  # [B, 3, C]
        virtual_coord = virtual_coord + agg
        return virtual_coord
    

    def node_model(self, node_feat, edge_index, edge_feat, virtual_edge_feat, node_attr):
        # node_feat: [batch_node, H], edge_feat: [batch_edge, H], virtual_edge_feat: [batch_node, H, C]
        row, col = edge_index
        agg = unsorted_segment_mean(edge_feat, row, num_segments=node_feat.size(0))  # [batch_node, H]
        virtual_edge_feat = virtual_edge_feat.reshape(virtual_edge_feat.size(0), -1)
        if node_attr is not None:
            agg = torch.cat([node_feat, agg, virtual_edge_feat, node_attr], dim=1)
        else:
            agg = torch.cat([node_feat, agg, virtual_edge_feat], dim=1)
        out = self.node_mlp(agg)

        if self.residual:
            out = node_feat + out
        return out

    def node_model_virtual(self, virtual_node_feat, virtual_edge_feat, data_batch):
        # virtual_node_feat: [B, H, C], virtual_edge_feat: [batch_node, H, C]
        agg = global_mean_pool(virtual_edge_feat.reshape(virtual_edge_feat.size(0), -1), data_batch) \
              .reshape(-1, self.hiddden_nf, self.virtual_channels)  # [B, H, C]
        out = torch.cat([virtual_node_feat, agg], dim=1)
        out = self.node_mlp_virtual(out.permute(0, 2, 1)).permute(0, 2, 1)
        
        if self.residual:
            out = virtual_node_feat + out
        return out


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff


    def forward(self, node_feat, edge_index, coord, node_vel, virtual_coord, virtual_node_feat, data_batch, edge_attr=None, node_attr=None):
        '''
        :param node_feat: feature of real node [batch_node, H]
        :param edge_index: edge index [2, batch_edge]
        :param coord: coordinate of real node [batch_node, 3]
        :param node_vel: velocity of real node [batch_node, 3]
        :param virtual_coord: Channels of virtual node [B, 3, C]
        :param virtual_node_feat: feature of virutal node [B, H, C]
        :param data_batch: data['batch'] [batch_node]
        :param edge_attr: attribute of edges [batch_edge, edge_attr_nf]
        :param node_attr: attribute of real nodes [batch_node, node_attr_nf]
        '''
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        virtual_coord_diff = virtual_coord[data_batch] - coord.unsqueeze(-1)  # [batch_node, 3, C]  (X - x)
        vitrual_radial = torch.norm(virtual_coord_diff, p=2, dim=1, keepdim=True)  # [batch_node, 1, C]

        # Edge Model
        edge_feat = self.edge_model(node_feat[row], node_feat[col], radial, edge_attr)  # [batch_edge, H]
        
        coord_mean = global_mean_pool(coord, data_batch)  # [B, 3]
        m_X = virtual_coord - coord_mean.unsqueeze(-1)  # [B, 3, C]
        m_X = torch.einsum('bij, bjk -> bik', m_X.permute(0, 2, 1), m_X)  # [B, C, C]
        # [batch_node, H, 1]  [batch_node, H, C], [batch_node, 1, C], [batch_node, C, C] -> [batch_node, H, C]
        virtual_edge_feat = self.edge_mode_virtual(node_feat.unsqueeze(-1), virtual_node_feat[data_batch], vitrual_radial, m_X[data_batch])  # [batch_edge, H, C]
        
        # Update by SchNet
        coord, _ = self.SchNetLayer(z=node_feat, pos=coord, batch=data_batch, edge_index=edge_index)
        
        # Coord Model
        coord = self.coord_model(node_feat, coord, node_vel, edge_index, coord_diff, edge_feat, virtual_edge_feat, virtual_coord_diff)
        virtual_coord = self.coord_model_virtual(virtual_coord, virtual_edge_feat, virtual_coord_diff, data_batch)
        
        # Node Model
        node_feat = self.node_model(node_feat, edge_index, edge_feat, virtual_edge_feat, node_attr)
        virtual_node_feat = self.node_model_virtual(virtual_node_feat, virtual_edge_feat, data_batch)
        return node_feat, coord, virtual_node_feat, virtual_coord


class FastSchNet(nn.Module):
    def __init__(self, node_feat_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels, device='cpu', act_fn=nn.SiLU(), 
                 n_layers=4, residual=True, attention=False, normalize=False, tanh=False, gravity=None):
        '''
        :param node_feat_nf: Number of node features
        :param node_attr_nf: Number of node attributes
        :param edge_attr_nf: Number of edge attributes
        :param hidden_nf: Number of hidden features
        :param virtual_channels: Channels of virtual node
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(FastSchNet, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.virtual_channels = virtual_channels
        assert virtual_channels > 0, f'Channels of virtual node must greater than 0 (got {virtual_channels})'
        self.virtual_node_feat = nn.Parameter(data=torch.randn(size=(1, hidden_nf, virtual_channels)), requires_grad=True)
        self.W = nn.Parameter(data=torch.randn(size=(1, virtual_channels, 3), requires_grad=True))
        self.embedding_in = nn.Linear(node_feat_nf, self.hidden_nf)
        if gravity is not None:
            gravity = torch.tensor(gravity, device=device)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, SchNet_GCL(hidden_nf, hidden_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels=virtual_channels,
                                                    act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh, gravity=gravity))
        self.to(self.device)

    def forward(self, node_feat, node_loc, node_vel, edge_index, data_batch, loc_mean, edge_attr=None, node_attr=None):
        # init virtual node feat with multi-channels
        batch_size = data_batch[-1].item() + 1
        virtual_node_feat = self.virtual_node_feat.repeat(batch_size, 1, 1)
        virtual_node_loc  = loc_mean

        node_feat = self.embedding_in(node_feat)
        for i in range(0, self.n_layers):
            node_feat, node_loc, virtual_node_feat, virtual_node_loc = \
                  self._modules["gcl_%d" % i](node_feat, edge_index, node_loc, node_vel, virtual_node_loc, virtual_node_feat, 
                                                data_batch, edge_attr=edge_attr, node_attr=node_attr)
            # print(f'layer: {i + 1}, node_loc: {node_loc}')
        return node_loc, virtual_node_loc


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)



"""
    Defination of SchNet from PyG
"""
class SchNet(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        interaction_graph (callable, optional): The function used to compute
            the pairwise interaction graph and interatomic distances. If set to
            :obj:`None`, will construct a graph based on :obj:`cutoff` and
            :obj:`max_num_neighbors` properties.
            If provided, this method takes in :obj:`pos` and :obj:`batch`
            tensors and should return :obj:`(edge_index, edge_weight)` tensors.
            (default :obj:`None`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (str, optional): Whether to apply :obj:`"add"` or :obj:`"mean"`
            global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = 'add',
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: OptTensor = None,
        device='cpu',
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.sum_aggr = SumAggregation()
        self.readout = aggr_resolver('sum' if self.dipole else readout)
        self.mean = mean
        self.std = std
        self.scale = None

        if self.dipole:
            import ase

            atomic_mass = torch.from_numpy(ase.data.atomic_masses)
            self.register_buffer('atomic_mass', atomic_mass)

        # # Support z == 0 for padding atoms so that their embedding vectors
        # # are zeroed and do not receive any gradients.
        # self.embedding = Embedding(100, hidden_channels, padding_idx=0)
        
        # Change for use embedding in FastSchNet 
        # self.embedding = Linear(2, hidden_channels)
        # self.embedding = Linear(hidden_channels, hidden_channels)

        # if interaction_graph is not None:
        #     self.interaction_graph = interaction_graph
        # else:
        #     self.interaction_graph = RadiusInteractionGraph(
        #         cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)
        
        self.coord_updates = ModuleList()
        for _ in range(num_interactions):
            block = Linear(num_gaussians + 2 * hidden_channels, 1)
            self.coord_updates.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        # self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z: Tensor, pos: Tensor, edge_index,
                batch: OptTensor = None) -> Tensor:
        r"""
        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        batch = torch.zeros_like(z) if batch is None else batch

        # h = self.embedding(z)
        h = z
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        # edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        row, col = edge_index
        
        for interaction, coord_update in zip(self.interactions, self.coord_updates):
            aggr = (pos[row] - pos[col]) * coord_update(torch.cat([edge_attr, h[row], h[col]], dim=-1))
            # aggr[index[i][j]][j] += data[i][j]
            aggr = scatter_mean(aggr, index=row.unsqueeze(-1).repeat(1, 3), dim=0)
            # print(aggr.size(), pos.size())
            pos[:aggr.size(0), :] += aggr
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        
        return pos, h

        # h = self.lin1(h)
        # h = self.act(h)
        # h = self.lin2(h)

        # if self.dipole:
        #     # Get center of mass.
        #     mass = self.atomic_mass[z].view(-1, 1)
        #     M = self.sum_aggr(mass, batch, dim=0)
        #     c = self.sum_aggr(mass * pos, batch, dim=0) / M
        #     h = h * (pos - c.index_select(0, batch))

        # if not self.dipole and self.mean is not None and self.std is not None:
        #     h = h * self.std + self.mean

        # if not self.dipole and self.atomref is not None:
        #     h = h + self.atomref(z)

        # out = self.readout(h, batch, dim=0)

        # if self.dipole:
        #     out = torch.norm(out, dim=-1, keepdim=True)

        # if self.scale is not None:
        #     out = self.scale * out

        # return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


class RadiusInteractionGraph(torch.nn.Module):
    r"""Creates edges based on atom positions :obj:`pos` to all points within
    the cutoff distance.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance with the
            default interaction graph method.
            (default: :obj:`32`)
    """
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            pos (Tensor): Coordinates of each atom.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
    ):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift