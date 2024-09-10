import torch
import functools
import torch.nn.functional as F

from torch import nn
from torch_sparse import spmm
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing


"""
    Basic models:
        EGNN
        EGMN
        EGHN
        GNN
        Linear
        RF

        GVP
        TFN
        SchNet
        DimeNet
"""


def aggregate(message, row_index, n_node, aggr='sum', mask=None):
    """
    The aggregation function (aggregate edge messages towards nodes)
    :param message: The edge message with shape [M, K]
    :param row_index: The row index of edges with shape [M]
    :param n_node: The number of nodes, N
    :param aggr: aggregation type, sum or mean
    :param mask: the edge mask (used in mean aggregation for counting degree)
    :return: The aggreagated node-wise information with shape [N, K]
    """
    result_shape = (n_node, message.shape[1])
    result = message.new_full(result_shape, 0)  # [N, K]
    row_index = row_index.unsqueeze(-1).expand(-1, message.shape[1])  # [M, K]
    result.scatter_add_(0, row_index, message)  # [N, K]
    if aggr == 'sum':
        pass
    elif aggr == 'mean':
        count = message.new_full(result_shape, 0)
        ones = torch.ones_like(message)
        if mask is not None:
            ones = ones * mask.unsqueeze(-1)
        count.scatter_add_(0, row_index, ones)
        result = result / count.clamp(min=1)
    else:
        raise NotImplementedError('Unknown aggregation method:', aggr)
    return result  # [N, K]


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
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


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True,
                 coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg*self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff/(norm + 1)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, residual=False, last_act=False, flat=False):
        super(BaseMLP, self).__init__()
        self.residual = residual
        if flat:
            activation = nn.Tanh()
            hidden_dim = 4 * hidden_dim
        if residual:
            assert output_dim == input_dim
        if last_act:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim),
                activation
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        return self.mlp(x) if not self.residual else self.mlp(x) + x


class EquivariantScalarNet(nn.Module):
    def __init__(self, n_vector_input, hidden_dim, activation, n_scalar_input=0, norm=True, flat=True):
        """
        The universal O(n) equivariant network using scalars.
        :param n_input: The total number of input vectors.
        :param hidden_dim: The hidden dim of the network.
        :param activation: The activation function.
        """
        super(EquivariantScalarNet, self).__init__()
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        # self.output_dim = n_vector_input
        self.activation = activation
        self.norm = norm
        self.in_scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.hidden_dim, self.activation, last_act=True,
                                     flat=flat)
        self.out_vector_net = BaseMLP(self.hidden_dim, self.hidden_dim, n_vector_input, self.activation, flat=flat)
        self.out_scalar_net = BaseMLP(self.hidden_dim, self.hidden_dim, self.output_dim, self.activation, flat=flat)

    def forward(self, vectors, scalars=None):
        """
        :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor
        :param scalars: torch.Tensor with shape [N, L] (Optional)
        :return: A vector that is equivariant to the O(n) transformations of input vectors with shape [N, 3]
        """
        if type(vectors) == list:
            Z = torch.stack(vectors, dim=-1)  # [N, 3, K]
        else:
            Z = vectors
        K = Z.shape[-1]
        Z_T = Z.transpose(-1, -2)  # [N, K, 3]
        scalar = torch.einsum('bij,bjk->bik', Z_T, Z)  # [N, K, K]
        scalar = scalar.reshape(-1, K * K)  # [N, KK]
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, KK]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, KK + L]
        scalar = self.in_scalar_net(scalar)  # [N, K]
        vec_scalar = self.out_vector_net(scalar)  # [N, K]
        vector = torch.einsum('bij,bj->bi', Z, vec_scalar)  # [N, 3]
        scalar = self.out_scalar_net(scalar)  # [N, H]

        return vector, scalar


class InvariantScalarNet(nn.Module):
    def __init__(self, n_vector_input, hidden_dim, output_dim, activation, n_scalar_input=0, norm=True, last_act=False,
                 flat=False):
        """
        The universal O(n) invariant network using scalars.
        :param n_vector_input: The total number of input vectors.
        :param hidden_dim: The hidden dim of the network.
        :param activation: The activation function.
        """
        super(InvariantScalarNet, self).__init__()
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.norm = norm
        self.scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.output_dim, self.activation, last_act=last_act,
                                  flat=flat)

    def forward(self, vectors, scalars=None):
        """
        :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor with shape [N, 3]
        :param scalars: torch.Tensor with shape [N, L] (Optional)
        :return: A scalar that is invariant to the O(n) transformations of input vectors  with shape [N, K]
        """
        if type(vectors) == list:
            Z = torch.stack(vectors, dim=-1)  # [N, 3, K]
        else:
            Z = vectors
        K = Z.shape[-1]
        Z_T = Z.transpose(-1, -2)  # [N, K, 3]
        scalar = torch.einsum('bij,bjk->bik', Z_T, Z)  # [N, K, K]
        scalar = scalar.reshape(-1, K * K)  # [N, KK]
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, KK]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, KK + L]
        scalar = self.scalar_net(scalar)  # [N, K]
        return scalar


class EGNN_Layer(nn.Module):
    def __init__(self, in_edge_nf, hidden_nf, activation=nn.SiLU(), with_v=False, flat=False, norm=False):
        super(EGNN_Layer, self).__init__()
        self.with_v = with_v
        self.edge_message_net = InvariantScalarNet(n_vector_input=1, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                                   activation=activation, n_scalar_input=2 * hidden_nf + in_edge_nf,
                                                   norm=norm, last_act=True, flat=flat)
        self.coord_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
                                 flat=flat)
        self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                activation=activation, flat=flat)
        if self.with_v:
            self.node_v_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
                                      flat=flat)
        else:
            self.node_v_net = None

    def forward(self, x, h, edge_index, edge_fea, v=None):
        row, col = edge_index
        rij = x[row] - x[col]  # [BM, 3]
        hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [BM, 2K+T]
        message = self.edge_message_net(vectors=[rij], scalars=hij)  # [BM, 3]
        coord_message = self.coord_net(message)  # [BM, 1]
        f = (x[row] - x[col]) * coord_message  # [BM, 3]
        tot_f = aggregate(message=f, row_index=row, n_node=x.shape[0], aggr='mean')  # [BN, 3]
        tot_f = torch.clamp(tot_f, min=-100, max=100)

        if v is not None:
            x = x + self.node_v_net(h) * v + tot_f
        else:
            x = x + tot_f  # [BN, 3]

        tot_message = aggregate(message=message, row_index=row, n_node=x.shape[0], aggr='mean')  # [BN, K]
        node_message = torch.cat((h, tot_message), dim=-1)  # [BN, K+K]
        h = self.node_net(node_message)  # [BN, K]
        return x, v, h


class EGNN(nn.Module):
    def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), device='cpu', with_v=False,
                 flat=False, norm=False):
        super(EGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.with_v = with_v
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(self.n_layers):
            layer = EGNN_Layer(in_edge_nf, hidden_nf, activation=activation, with_v=with_v, flat=flat, norm=norm)
            self.layers.append(layer)
        self.to(device)

    def forward(self, x, h, edge_index, edge_fea, v=None):
        h = self.embedding(h)
        for i in range(self.n_layers):
            x, v, h = self.layers[i](x, h, edge_index, edge_fea, v=v)
        return (x, v, h) if v is not None else (x, h)


class EGMN(nn.Module):
    def __init__(self, n_layers, n_vector_input, hidden_dim, n_scalar_input, activation=nn.SiLU(), device='cpu', norm=False, flat=False):
        super(EGMN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(self.n_layers):
            cur_layer = EquivariantScalarNet(n_vector_input=n_vector_input + i, hidden_dim=hidden_dim,
                                             activation=activation, n_scalar_input=n_scalar_input if i == 0 else hidden_dim,
                                             norm=norm, flat=flat)
            self.layers.append(cur_layer)
        self.to(device)

    def forward(self, vectors, scalars):
        cur_vectors = vectors
        for i in range(self.n_layers):
            vector, scalars = self.layers[i](cur_vectors, scalars)
            cur_vectors.append(vector)
        return cur_vectors[-1], scalars


class GNN_Layer(nn.Module):
    def __init__(self, in_edge_nf, hidden_nf, activation=nn.SiLU(), with_v=False, flat=False):
        super(GNN_Layer, self).__init__()
        self.with_v = with_v
        self.edge_message_net = BaseMLP(input_dim=in_edge_nf + 2 * hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                        activation=activation, flat=flat)
        self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                activation=activation, flat=flat)

    def forward(self, h, edge_index, edge_fea):
        row, col = edge_index
        hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [BM, 2K+T]
        message = self.edge_message_net(hij)  # [BM, K]
        agg = aggregate(message=message, row_index=row, n_node=h.shape[0], aggr='mean')  # [BN, K]
        h = h + self.node_net(torch.cat((agg, h), dim=-1))
        return h


class GNN(nn.Module):
    def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), device='cpu', flat=False):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(self.n_layers):
            layer = GNN_Layer(in_edge_nf, hidden_nf, activation=activation, flat=flat)
            self.layers.append(layer)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, 3)
        )
        self.to(device)

    def forward(self, h, edge_index, edge_fea):
        h = self.embedding(h)
        for i in range(self.n_layers):
            h = self.layers[i](h, edge_index, edge_fea)
        h = self.decoder(h)
        return h


class Linear_dynamics(nn.Module):
    def __init__(self, device='cpu'):
        super(Linear_dynamics, self).__init__()
        self.time = nn.Parameter(torch.ones(1))
        self.device = device
        self.to(self.device)

    def forward(self, x, v):
        return x + v * self.time


class RF_vel(nn.Module):
    def __init__(self, hidden_nf, edge_attr_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4):
        super(RF_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL_rf_vel(nf=hidden_nf, edge_attr_nf=edge_attr_nf, act_fn=act_fn))
        self.to(self.device)

    def forward(self, vel_norm, x, edges, vel, edge_attr):
        for i in range(0, self.n_layers):
            x, _ = self._modules["gcl_%d" % i](x, vel_norm, vel, edges, edge_attr)
        return x


class GCL_rf_vel(nn.Module):
    def __init__(self,  nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0):
        super(GCL_rf_vel, self).__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, nf),
            act_fn,
            nn.Linear(nf, 1))

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(1 + edge_attr_nf, nf),
                                 act_fn,
                                 layer,
                                 nn.Tanh())

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x += vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        row, col = edge_index
        agg = aggregate(edge_m, row, n_node=x.size(0), aggr='mean')
        x_out = x + agg * self.coords_weight
        return x_out
    

class EquivariantEdgeScalarNet(nn.Module):
    def __init__(self, n_vector_input, hidden_dim, activation, n_scalar_input=0, norm=True, flat=False):
        """
        The universal O(n) equivariant network using scalars.
        :param n_input: The total number of input vectors.
        :param hidden_dim: The hidden dim of the network.
        :param activation: The activation function.
        """
        super(EquivariantEdgeScalarNet, self).__init__()
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        # self.output_dim = n_vector_input
        self.activation = activation
        self.norm = norm
        self.in_scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.hidden_dim, self.activation, last_act=True,
                                     flat=flat)
        self.out_vector_net = BaseMLP(self.hidden_dim, self.hidden_dim, n_vector_input * n_vector_input,
                                      self.activation, flat=flat)

    def forward(self, vectors_i, vectors_j, scalars=None):
        """
        :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor
        :param scalars: torch.Tensor with shape [N, L] (Optional)
        :return: A vector that is equivariant to the O(n) transformations of input vectors with shape [N, 3]
        """
        Z_i, Z_j = vectors_i, vectors_j  # [N, 3, K]
        K = Z_i.shape[-1]
        Z_j_T = Z_j.transpose(-1, -2)  # [N, K, 3]
        scalar = torch.einsum('bij,bjk->bik', Z_j_T, Z_i)  # [N, K, K]
        scalar = scalar.reshape(-1, K * K)  # [N, KK]
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, KK]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, KK + L]
        scalar = self.in_scalar_net(scalar)  # [N, H]
        vec_scalar = self.out_vector_net(scalar)  # [N, KK]
        vec_scalar = vec_scalar.reshape(-1, Z_j.shape[-1], Z_i.shape[-1])  # [N, K, K]
        vector = torch.einsum('bij,bjk->bik', Z_j, vec_scalar)  # [N, 3, K]
        return vector, scalar


class PoolingLayer(nn.Module):
    def __init__(self, in_edge_nf, hidden_nf, n_vector_input, activation=nn.SiLU(), flat=False):
        super(PoolingLayer, self).__init__()
        self.edge_message_net = EquivariantEdgeScalarNet(n_vector_input=n_vector_input, hidden_dim=hidden_nf,
                                                         activation=activation, n_scalar_input=2 * hidden_nf + in_edge_nf,
                                                         norm=True, flat=flat)
        self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                activation=activation, flat=flat)

    def forward(self, vectors, h, edge_index, edge_fea):
        """
        :param vectors: the node vectors with shape: [BN, 3, V] where V is the number of vectors
        :param h: the scalar node feature with shape: [BN, K]
        :param edge_index: the edge index with shape [2, BM]
        :param edge_fea: the edge feature with shape: [BM, T]
        :return: the updated node vectors [BN, 3, V] and node scalar feature [BN, K]
        """
        row, col = edge_index
        hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [BM, 2K+T]
        vectors_i, vectors_j = vectors[row], vectors[col]  # [BM, 3, V]
        vectors_out, message = self.edge_message_net(vectors_i=vectors_i, vectors_j=vectors_j, scalars=hij)  # [BM, 3, V]
        DIM, V = vectors_out.shape[-2], vectors_out.shape[-1]
        vectors_out = vectors_out.reshape(-1, DIM * V)  # [BM, 3V]
        vectors_out = aggregate(message=vectors_out, row_index=row, n_node=h.shape[0], aggr='mean')  # [BN, 3V]
        vectors_out = vectors_out.reshape(-1, DIM, V)  # [BN, 3, V]
        vectors_out = vectors + vectors_out  # [BN, 3, V]
        tot_message = aggregate(message=message, row_index=row, n_node=h.shape[0], aggr='sum')  # [BN, K]
        node_message = torch.cat((h, tot_message), dim=-1)  # [BN, K+K]
        h = self.node_net(node_message) + h  # [BN, K]
        return vectors_out, h


class PoolingNet(nn.Module):
    def __init__(self, n_layers, in_edge_nf, n_vector_input,
                 hidden_nf, output_nf, activation=nn.SiLU(), device='cpu', flat=False):
        super(PoolingNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(self.n_layers):
            layer = PoolingLayer(in_edge_nf, hidden_nf, n_vector_input=n_vector_input, activation=activation, flat=flat)
            self.layers.append(layer)
        self.pooling = nn.Sequential(
            nn.Linear(hidden_nf, 8 * hidden_nf),
            nn.Tanh(),
            nn.Linear(8 * hidden_nf, output_nf)
        )
        self.to(device)

    def forward(self, vectors, h, edge_index, edge_fea):
        if type(vectors) == list:
            vectors = torch.stack(vectors, dim=-1)  # [BN, 3, V]
        for i in range(self.n_layers):
            vectors, h = self.layers[i](vectors, h, edge_index, edge_fea)
        pooling = self.pooling(h)
        return pooling  # [BN, P]


class EGHN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, n_cluster, layer_per_block=3, layer_pooling=3, layer_decoder=1,
                 flat=False, activation=nn.SiLU(), device='cpu', norm=False, with_v=True):
        super(EGHN, self).__init__()
        node_hidden_dim = hidden_nf
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.current_pooling_plan = None
        self.n_cluster = n_cluster  # 4 for simulation and 5 for mocap
        self.n_layer_per_block = layer_per_block
        self.n_layer_pooling = layer_pooling
        self.n_layer_decoder = layer_decoder
        self.flat = flat
        self.with_v = with_v
        # low-level force net
        self.low_force_net = EGNN(n_layers=self.n_layer_per_block,
                                  in_node_nf=hidden_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf,
                                  activation=activation, device=device, with_v=with_v, flat=flat, norm=norm)
        self.low_pooling = PoolingNet(n_vector_input=3, hidden_nf=hidden_nf, output_nf=self.n_cluster,
                                      activation=activation, in_edge_nf=in_edge_nf, n_layers=self.n_layer_pooling, flat=flat)
        self.high_force_net = EGNN(n_layers=self.n_layer_per_block,
                                   in_node_nf=hidden_nf, in_edge_nf=1, hidden_nf=hidden_nf,
                                   activation=activation, device=device, with_v=with_v, flat=flat)
        _n_vector_input = 4 if self.with_v else 3
        if self.n_layer_decoder == 1:
            self.kinematics_net = EquivariantScalarNet(n_vector_input=_n_vector_input,
                                                       hidden_dim=hidden_nf,
                                                       activation=activation,
                                                       n_scalar_input=node_hidden_dim + node_hidden_dim,
                                                       norm=True,
                                                       flat=flat)
        else:
            self.kinematics_net = EGMN(n_vector_input=_n_vector_input, hidden_dim=hidden_nf, activation=activation,
                                       n_scalar_input=node_hidden_dim + node_hidden_dim, norm=True, flat=flat,
                                       n_layers=self.n_layer_decoder)

        self.to(device)

    def forward(self, x, h, edge_index, edge_fea, local_edge_index, local_edge_fea, n_node, v=None, node_mask=None, node_nums=None):
        """
        :param x: input positions [B * N, 3]
        :param h: input node feature [B * N, R]
        :param edge_index: edge index of the graph [2, B * M]
        :param edge_fea: input edge feature [B* M, T]
        :param local_edge_index: the edges used in pooling network [B * M']
        :param local_edge_fea: the feature of local edges [B * M', T]
        :param n_node: number of nodes per graph [1, ]
        :param v: input velocities [B * N, 3] (Optional)
        :param node_mask: the node mask when number of nodes are different in graphs [B * N, ] (Optional)
        :param node_nums: the real number of nodes in each graph
        :return:
        """
        h = self.embedding(h)  # [R, K]
        row, col = edge_index

        ''' low level force '''
        new_x, new_v, h = self.low_force_net(x, h, edge_index, edge_fea, v=v)  # [BN, 3]
        nf = new_x - x  # [BN, 3]

        ''' pooling network '''
        if node_nums is None:
            x_mean = torch.mean(x.reshape(-1, n_node, x.shape[-1]), dim=1, keepdim=True).expand(-1, n_node, -1).reshape(
                -1, x.shape[-1])
        else:
            pooled_mean = (torch.sum(x.reshape(-1, n_node, x.shape[-1]), dim=1).T/node_nums).T.unsqueeze(dim=1) #[B,1,3]
            x_mean = pooled_mean.expand(-1, n_node, -1).reshape(-1, x.shape[-1])

        pooling_fea = self.low_pooling(vectors=[x - x_mean, nf, v], h=h,
                                       edge_index=local_edge_index, edge_fea=local_edge_fea)  # [BN, P]

        hard_pooling = pooling_fea.argmax(dim=-1)
        hard_pooling = F.one_hot(hard_pooling, num_classes=self.n_cluster).float()
        pooling = F.softmax(pooling_fea, dim=1)
        self.current_pooling_plan = hard_pooling  # record the pooling plan

        ''' derive high-level information (be careful with graph mini-batch) '''
        s = pooling.reshape(-1, n_node, pooling.shape[-1])  # [B, N, P]

        sT = s.transpose(-2, -1)  # [B, P, N]
        p_index = torch.ones_like(nf)[..., 0]  # [BN, ]
        if node_mask is not None:
            p_index = p_index * node_mask
        p_index = p_index.reshape(-1, n_node, 1)  # [B, N, 1]
        count = torch.einsum('bij,bjk->bik', sT, p_index).clamp_min(1e-5)  # [B, P, 1]
        _x, _h, _nf = x.reshape(-1, n_node, x.shape[-1]), h.reshape(-1, n_node, h.shape[-1]), nf.reshape(-1, n_node, nf.shape[-1])
        # [B, N, 3], [B, N, K], [B, N, 3]
        X, H, NF = torch.einsum('bij,bjk->bik', sT, _x), torch.einsum('bij,bjk->bik', sT, _h), torch.einsum('bij,bjk->bik', sT, _nf)
        if v is not None:
            _v = v.reshape(-1, n_node, v.shape[-1])
            V = torch.einsum('bij,bjk->bik', sT, _v)
            V = V / count
            V = V.reshape(-1, V.shape[-1])
        else:
            V = None
        X, H, NF = X / count, H / count, NF / count  # [B, P, 3], [B, P, K], [B, P, 3]
        X, H, NF = X.reshape(-1, X.shape[-1]), H.reshape(-1, H.shape[-1]), NF.reshape(-1, NF.shape[-1])  # [BP, 3]

        a = spmm(torch.stack((local_edge_index[0], local_edge_index[1]), dim=0),
                 torch.ones_like(local_edge_index[0]), x.shape[0], x.shape[0], pooling)  # [BN, P]
        a = a.reshape(-1, n_node, a.shape[-1])  # [B, N, P]
        A = torch.einsum('bij,bjk->bik', sT, a)  # [B, P, P]
        self.cut_loss = self.get_cut_loss(A)
        aa = spmm(torch.stack((row, col), dim=0), torch.ones_like(row), x.shape[0], x.shape[0], pooling)  # [BN, P]
        aa = aa.reshape(-1, n_node, aa.shape[-1])  # [B, N, P]
        AA = torch.einsum('bij,bjk->bik', sT, aa)  # [B, P, P]

        # construct high-level edges
        h_row, h_col, h_edge_fea, h_edge_mask = self.construct_edges(AA, AA.shape[-1])  # [BPP]
        ''' high-level message passing '''
        h_new_x, h_new_v, h_new_h = self.high_force_net(X, H, (h_row, h_col), h_edge_fea.unsqueeze(-1), v=V)
        h_nf = h_new_x - X

        ''' high-level kinematics update '''
        _X = X + h_nf  # [BP, 3]
        _V = h_new_v  # [BP, 3]
        _H = h_new_h  # [BP, K]

        ''' low-level kinematics update '''
        l_nf = h_nf.reshape(-1, AA.shape[1], h_nf.shape[-1])  # [B, P, 3]
        l_nf = torch.einsum('bij,bjk->bik', s, l_nf).reshape(-1, l_nf.shape[-1])  # [BN, 3]
        l_X = X.reshape(-1, AA.shape[1], X.shape[-1])  # [B, P, 3]
        l_X = torch.einsum('bij,bjk->bik', s, l_X).reshape(-1, l_X.shape[-1])  # [BN, 3]
        if v is not None:
            l_V = V.reshape(-1, AA.shape[1], V.shape[-1])  # [B, P, 3]
            l_V = torch.einsum('bij,bjk->bik', s, l_V).reshape(-1, l_V.shape[-1])  # [BN, 3]
            vectors = [l_nf, x - l_X, v - l_V, nf]
        else:
            vectors = [l_nf, x - l_X, nf]
        l_H = _H.reshape(-1, AA.shape[1], _H.shape[-1])  # [B, P, K]
        l_H = torch.einsum('bij,bjk->bik', s, l_H).reshape(-1, l_H.shape[-1])  # [BN, K]

        l_kinematics, h_out = self.kinematics_net(vectors=vectors,
                                                  scalars=torch.cat((h, l_H), dim=-1))  # [BN, 3]
        _l_X = _X.reshape(-1, AA.shape[1], _X.shape[-1])  # [B, P, 3]
        _l_X = torch.einsum('bij,bjk->bik', s, _l_X).reshape(-1, _l_X.shape[-1])  # [BN, 3]
        x_out = _l_X + l_kinematics  # [BN, 3]

        return (x_out, v, h_out) if v is not None else (x_out, h_out)

    def inspect_pooling_plan(self):
        plan = self.current_pooling_plan  # [BN, P]
        if plan is None:
            print('No pooling plan!')
            return
        dist = torch.sum(plan, dim=0)  # [P,]
        # print(dist)
        dist = F.normalize(dist, p=1, dim=0)  # [P,]
        print('Pooling plan:', dist.detach().cpu().numpy())
        return

    def get_cut_loss(self, A):
        A = F.normalize(A, p=2, dim=2)
        return torch.norm(A - torch.eye(A.shape[-1]).to(A.device), p="fro", dim=[1, 2]).mean()

    @staticmethod
    def construct_edges(A, n_node):
        h_edge_fea = A.reshape(-1)  # [BPP]
        h_row = torch.arange(A.shape[1]).unsqueeze(-1).expand(-1, A.shape[1]).reshape(-1).to(A.device)
        h_col = torch.arange(A.shape[1]).unsqueeze(0).expand(A.shape[1], -1).reshape(-1).to(A.device)
        h_row = h_row.unsqueeze(0).expand(A.shape[0], -1)
        h_col = h_col.unsqueeze(0).expand(A.shape[0], -1)
        offset = (torch.arange(A.shape[0]) * n_node).unsqueeze(-1).to(A.device)
        h_row, h_col = (h_row + offset).reshape(-1), (h_col + offset).reshape(-1)  # [BPP]
        h_edge_mask = torch.ones_like(h_row)  # [BPP]
        h_edge_mask[torch.arange(A.shape[1]) * (A.shape[1] + 1)] = 0
        return h_row, h_col, h_edge_fea, h_edge_mask


class FullMLP(nn.ModuleList):
    def __init__(self, in_node_nf, hidden_nf, n_layers, activation=nn.SiLU(), flat=False, device='cpu'):
        super(FullMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(n_layers):
            self.layers.append(BaseMLP(hidden_nf, hidden_nf, hidden_nf, activation,
                                       residual=True, last_act=True, flat=flat))
        self.output = nn.Linear(hidden_nf, 3)
        self.to(device)

    def forward(self, x):
        x = self.embedding(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return self.output(x)

