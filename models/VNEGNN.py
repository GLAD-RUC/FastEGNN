import torch
from torch import nn

from torch_geometric.nn import global_mean_pool

from torch_scatter import scatter_add


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



class EGCL_A2A(nn.Module):
    def __init__(self, node_feat_nf, node_feat_out_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels,
                 act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(EGCL_A2A, self).__init__()
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
        self.virtual_channels = virtual_channels  # no use in A2A

        # MLPS
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_feat_nf + edge_coords_nf + edge_attr_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + hidden_nf + node_attr_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
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
        if self.residual:
            out = x + out
        return out, agg
    

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord
    

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff
    

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord



class EGCL_A2V(nn.Module):
    def __init__(self, node_feat_nf, node_feat_out_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels,
                 act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(EGCL_A2V, self).__init__()
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

        # MLPS
        self.edge_mlp = nn.Sequential(  # OK
            nn.Linear(2 * node_feat_nf + edge_coords_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:  # OK
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())


    # [batch_node, H, 1]  [batch_node, H, C], [batch_node, 1, C] -> [batch_node, H, C]
    def edge_model_A2V(self, feat_R, feat_V, radial):
        feat_R = feat_R.repeat(1, 1, self.virtual_channels)  # [batch_size, H, C]

        out = torch.cat([feat_R, feat_V, radial], dim=1)
        out = self.edge_mlp(out.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, C, H]
        if self.attention:
            att_val = self.att_mlp(out.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, C, 1]
            out = out * att_val  # [batch_node, C, H]
        return out
    
    
    def coord_model_A2V(self, virtual_coord, edge_feat, virtual_coord_diff, data_batch):
        trans = virtual_coord_diff * self.coord_mlp(edge_feat.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, 3, C]
        agg = global_mean_pool(trans.reshape(trans.size(0), -1), data_batch).reshape(-1, 3, self.virtual_channels)  # [B, 3, C]
        virtual_coord += agg
        return virtual_coord
    
    
    def node_model_A2V(self, virtual_node_feat, virtual_edge_feat, data_batch):
        # virtual_node_feat: [B, H, C], virtual_edge_feat: [batch_node, H, C]
        agg = global_mean_pool(virtual_edge_feat.reshape(virtual_edge_feat.size(0), -1), data_batch) \
              .reshape(-1, self.hiddden_nf, self.virtual_channels)  # [B, H, C]
        out = torch.cat([virtual_node_feat, agg], dim=1)
        out = self.node_mlp(out.permute(0, 2, 1)).permute(0, 2, 1)
        
        if self.residual:
            out = virtual_node_feat + out
        return out


    def forward(self, node_feat, coord, virtual_node_feat, virtual_coord, data_batch):
        # node_feat: [batch_node, H]
        # coord: [batch_node, 3]
        # virtual_node_feat: [B, H, C]
        # virtual_node_coord: [B, 3, C]
        
        virtual_coord_diff = virtual_coord[data_batch] - coord.unsqueeze(-1)  # [batch_node, 3, C]  (X - x)
        vitrual_radial = torch.norm(virtual_coord_diff, p=2, dim=1, keepdim=True)  # [batch_node, 1, C]

        edge_feat_A2V = self.edge_model_A2V(node_feat.unsqueeze(-1), virtual_node_feat[data_batch], vitrual_radial)  # [batch_node, C, H] (every R to every V)
        virtual_coord = self.coord_model_A2V(virtual_coord, edge_feat_A2V, virtual_coord_diff, data_batch)  # [B, 3, C]
        virtual_node_feat = self.node_model_A2V(virtual_node_feat, edge_feat_A2V, data_batch)

        return virtual_node_feat, virtual_coord
    

class EGCL_V2A(nn.Module):
    def __init__(self, node_feat_nf, node_feat_out_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels,
                 act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(EGCL_V2A, self).__init__()
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

        # MLPS
        self.edge_mlp = nn.Sequential(  # OK
            nn.Linear(2 * node_feat_nf + edge_coords_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        
        self.node_mlp = nn.Sequential(  # \phi_{h}
            nn.Linear(hidden_nf + hidden_nf + node_attr_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, node_feat_out_nf)
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:  # OK
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())


    # [batch_node, H, 1]  [batch_node, H, C], [batch_node, 1, C] -> [batch_node, H, C]
    def edge_model_V2A(self, feat_R, feat_V, radial):
        feat_R = feat_R.repeat(1, 1, self.virtual_channels)  # [batch_size, H, C]

        out = torch.cat([feat_R, feat_V, radial], dim=1)
        out = self.edge_mlp(out.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, C, H]
        if self.attention:
            att_val = self.att_mlp(out.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, C, 1]
            out = out * att_val  # [batch_node, C, H]
        return out
    

    def coord_model_V2A(self, coord, virtual_edge_feat, virtual_coord_diff):
        # virtual_edge_feat: [batch_node, H, C], virtual_coord_diff: [batch_node, 3, C]
        trans_v = torch.mean(-virtual_coord_diff * self.coord_mlp(virtual_edge_feat.permute(0, 2, 1)).permute(0, 2, 1), dim=-1)  # [batch_node, 3]
        coord += trans_v
        return coord


    def node_model_V2A(self, node_feat, virtual_edge_feat, node_attr):
        # node_feat: [batch_node, H], virtual_edge_feat: [batch_node, H, C]
        virtual_edge_feat = torch.mean(virtual_edge_feat, dim=-1)  # [batch_node, H]
        # virtual_edge_feat = virtual_edge_feat.reshape(virtual_edge_feat.size(0), -1)  # different from FastEGNN

        if node_attr is not None:
            agg = torch.cat([node_feat, virtual_edge_feat, node_attr], dim=1)
        else:
            agg = torch.cat([node_feat, virtual_edge_feat], dim=1)
        out = self.node_mlp(agg)

        if self.residual:
            out = node_feat + out
        return out


    def forward(self, virtual_node_feat, virtual_coord, node_feat, coord, data_batch):
        virtual_coord_diff = virtual_coord[data_batch] - coord.unsqueeze(-1)  # [batch_node, 3, C]  (X - x)
        vitrual_radial = torch.norm(virtual_coord_diff, p=2, dim=1, keepdim=True)  # [batch_node, 1, C]

        edge_feat_V2A = self.edge_model_V2A(node_feat.unsqueeze(-1), virtual_node_feat[data_batch], vitrual_radial)  # [batch_node, C, H] (every R to every V)

        coord = self.coord_model_V2A(coord, edge_feat_V2A, virtual_coord_diff)
        node_feat = self.node_model_V2A(node_feat, edge_feat_V2A, node_attr=None)

        return node_feat, coord


class VNEGNN(nn.Module):
    def __init__(self, node_feat_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels, device='cpu', act_fn=nn.SiLU(), 
                 n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        super(VNEGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.virtual_channels = virtual_channels
        assert virtual_channels > 0, f'Channels of virtual node must greater than 0 (got {virtual_channels})'
        self.virtual_node_feat = nn.Parameter(data=torch.randn(size=(1, hidden_nf, virtual_channels)), requires_grad=True)

        self.embedding_in = nn.Linear(node_feat_nf, self.hidden_nf)

        for i in range(n_layers):
            self.add_module(f'A2A_{i}', EGCL_A2A(hidden_nf, hidden_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels=virtual_channels,
                                                act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh))
            self.add_module(f'A2V_{i}', EGCL_A2V(hidden_nf, hidden_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels=virtual_channels,
                                                act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh))
            self.add_module(f'V2A_{i}', EGCL_V2A(hidden_nf, hidden_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels=virtual_channels,
                                                act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh))
        
        
        self.to(self.device)

    def forward(self, node_feat, node_loc, edge_index, data_batch, virtual_node_loc, edge_attr=None, node_attr=None):
        # init virtual node feat with multi-channels
        batch_size = data_batch[-1].item() + 1
        virtual_node_feat = self.virtual_node_feat.repeat(batch_size, 1, 1)
        virtual_node_loc  = virtual_node_loc

        node_feat = self.embedding_in(node_feat)

        for i in range(self.n_layers):
            node_feat, node_loc = self._modules[f'A2A_{i}'](h=node_feat, edge_index=edge_index, coord=node_loc, edge_attr=edge_attr, node_attr=node_attr)
            virtual_node_feat, virtual_node_loc = self._modules[f'A2V_{i}'](node_feat, node_loc, virtual_node_feat, virtual_node_loc, data_batch=data_batch)
            node_feat, node_loc = self._modules[f'V2A_{i}'](virtual_node_feat, virtual_node_loc, node_feat, node_loc, data_batch=data_batch)

        return node_loc, virtual_node_loc
    