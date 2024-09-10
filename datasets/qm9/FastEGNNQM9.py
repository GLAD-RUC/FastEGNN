import torch
import torch_geometric
from torch import nn
from torch_geometric.nn import global_mean_pool

from torch_scatter import scatter_add

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, node_feat_nf, node_feat_out_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels,
                 act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, gravity=None):
        super(E_GCL, self).__init__()
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


    def coord_model_vel(self, node_feat, coord, edge_index, coord_diff, edge_feat, virtual_edge_feat, virtual_coord_diff):
        row, col = edge_index

        trans = coord_diff * self.coord_mlp_r(edge_feat)  # coord_mlp_r: [batch_edge, H] -> [batch_edge, 1]
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        
        coord += agg

        # virtual_edge_feat: [batch_node, H, C], virtual_coord_diff: [batch_node, 3, C]
        trans_v = torch.mean(-virtual_coord_diff * self.coord_mlp_r_virtual(virtual_edge_feat.permute(0, 2, 1)).permute(0, 2, 1), dim=-1)  # [batch_node, 3]
        coord += trans_v

        if self.gravity is not None:
            coord += self.gravity_mlp(node_feat) * self.gravity  # Gravity

        return coord
    
    def coord_model_virtual(self, virtual_coord, virtual_edge_feat, virtual_coord_diff, data_batch):
        trans = virtual_coord_diff * self.coord_mlp_v_virtual(virtual_edge_feat.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, 3, C]
        agg = global_mean_pool(trans.reshape(trans.size(0), -1), data_batch).reshape(-1, 3, self.virtual_channels)  # [B, 3, C]
        virtual_coord += agg
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


    def forward(self, node_feat, edge_index, coord, virtual_coord, virtual_node_feat, data_batch, edge_attr=None, node_attr=None):
        '''
        :param node_feat: feature of real node [batch_node, H]
        :param edge_index: edge index [2, batch_edge]
        :param coord: coordinate of real node [batch_node, 3]
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
        # Coord Model
        coord = self.coord_model_vel(node_feat, coord, edge_index, coord_diff, edge_feat, virtual_edge_feat, virtual_coord_diff)
        virtual_coord = self.coord_model_virtual(virtual_coord, virtual_edge_feat, virtual_coord_diff, data_batch)
        # Node Model
        node_feat = self.node_model(node_feat, edge_index, edge_feat, virtual_edge_feat, node_attr)
        virtual_node_feat = self.node_model_virtual(virtual_node_feat, virtual_edge_feat, data_batch)
        return node_feat, coord, virtual_node_feat, virtual_coord


class FastEGNNQM9(nn.Module):
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

        super(FastEGNNQM9, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.virtual_channels = virtual_channels
        assert virtual_channels > 0, f'Channels of virtual node must greater than 0 (got {virtual_channels})'
        self.virtual_node_feat = nn.Parameter(data=torch.randn(size=(1, hidden_nf, virtual_channels)), requires_grad=True)
        self.embedding_in = nn.Linear(node_feat_nf, self.hidden_nf)
        if gravity is not None:
            gravity = torch.tensor(gravity, device=device)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(hidden_nf, hidden_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels=virtual_channels,
                                                act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh, gravity=gravity))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))

        self.to(self.device)


    def forward(self, node_feat, node_loc, edge_index, data_batch, loc_mean, edge_attr=None, node_attr=None):
        # init virtual node feat with multi-channels
        batch_size = data_batch[-1].item() + 1
        virtual_node_feat = self.virtual_node_feat.repeat(batch_size, 1, 1)
        virtual_node_loc  = loc_mean.repeat(1, 1, self.virtual_channels)

        node_feat = self.embedding_in(node_feat)
        for i in range(0, self.n_layers):
            node_feat, node_loc, virtual_node_feat, virtual_node_loc = \
                  self._modules["gcl_%d" % i](node_feat, edge_index, node_loc, virtual_node_loc, virtual_node_feat, 
                                                data_batch, edge_attr=edge_attr, node_attr=node_attr)
        
        node_feat = self.node_dec(node_feat)
        graph_feat = scatter_add(node_feat, index=data_batch, dim=0)
        pred = self.graph_dec(graph_feat)
        return pred.squeeze(1)


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
