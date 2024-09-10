import torch

def compute_mean_mad(dataloaders, label_property):
    values = []
    for data in dataloaders['train']:
        values.append(data[label_property])
    # values = dataloaders['train'].dataset.data[label_property]
    values = torch.cat(values, dim=-1)
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

# one_hot: [1737, 5], charges: [1737]
def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_scale = charge_scale.to(device)
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))  # [1737, 3]
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))  # [1737, 1, 3]
    
    # [1737, 5, 1]
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))  # [1737, 15]
    return atom_scalars