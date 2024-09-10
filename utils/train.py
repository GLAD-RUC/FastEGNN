import os
import time
import json

import torch

from datasets.protein.dataset import MDAnalysisDataset

def get_edges_in_mini_batch(batch_size, num_nodes_all, edge_index):
    correction_for_batch = num_nodes_all * torch.arange(batch_size, device=edge_index.device)  # [batch_size]
    correction_for_batch = correction_for_batch.repeat_interleave(edge_index.size(1) // batch_size, dim=0).unsqueeze(0)  # [1, edge_cnt]
    correction_for_batch = correction_for_batch.repeat_interleave(2, dim=0)
    edge_index_in_mini_batch = edge_index + correction_for_batch
    return edge_index_in_mini_batch


def kernel(x, y, sigma):
    dist = torch.cdist(x, y, p=2)
    k = torch.exp(- dist / (2 * sigma * sigma))
    return k


def train_single_epoch(model, loader, optimizer, loss, sigma, weight, epoch_index, backprop, tag, sample, device='cpu', config=None):
    if backprop:
        model.train()
    else:
        model.eval()

    result = {'loss': 0., 'counter': 0.}
    for batch_index, data in enumerate(loader):
        # All to device
        data = data.to(device)
        data = data.detach()  # All detach

        # Parse data
        batch_size = data['ptr'].size(0) - 1
        edge_index, edge_attr = data['edge_index'], data['edge_attr']
        loc_0, vel_0, loc_t = data['loc_0'], data['vel_0'], data['loc_t']
        node_feat, node_attr = data['node_feat'], data['node_attr']

        row, col = edge_index
        edge_length_0 = torch.sqrt(torch.sum((loc_0[row] - loc_0[col])**2, dim=1)).unsqueeze(1)
        edge_attr = torch.cat([edge_attr, edge_length_0], dim=1)
        
        # detach from compute graph
        loc_0, vel_0, node_attr, node_feat = loc_0.detach(), vel_0.detach(), node_attr.detach(), node_feat.detach()
        edge_attr, edge_index = edge_attr.detach(), edge_index.detach()

        optimizer.zero_grad()

        if model.__class__.__name__ == 'FastEGNN':
            loc_predict, virtual_node_loc = model(node_loc=loc_0, node_vel=vel_0, node_attr=None, node_feat=node_feat, edge_index=edge_index, 
                                                  loc_mean=data['loc_mean'].detach(), data_batch=data['batch'], edge_attr=edge_attr)
        elif model.__class__.__name__ == 'VNEGNN':
            loc_predict, virtual_node_loc = model(node_loc=loc_0, node_attr=None, node_feat=node_feat, edge_index=edge_index, 
                                                  virtual_node_loc=data['virtual_fibonacci'].detach(), data_batch=data['batch'], edge_attr=edge_attr)
        elif model.__class__.__name__ == 'FastRF':
            loc_predict, virtual_node_loc = model(node_loc=loc_0, node_vel=vel_0, node_attr=None, node_feat=node_feat, edge_index=edge_index, 
                                                  loc_mean=data['loc_mean'].detach(), data_batch=data['batch'], edge_attr=edge_attr)
        elif model.__class__.__name__ == 'FastTFN':
            loc_predict, virtual_node_loc = model(node_loc=loc_0, node_vel=vel_0, node_attr=None, node_feat=node_feat, edge_index=edge_index, 
                                                  loc_mean=data['loc_mean'].detach(), data_batch=data['batch'], edge_attr=edge_attr, charges=node_attr)
        elif model.__class__.__name__ == 'FastSchNet':
            loc_predict, virtual_node_loc = model(node_loc=loc_0, node_vel=vel_0, node_attr=None, node_feat=node_feat, edge_index=edge_index, 
                                                  loc_mean=data['loc_mean'].detach(), data_batch=data['batch'], edge_attr=edge_attr)
        elif model.__class__.__name__ == 'EGNN':
            out = model(x=loc_0, h=node_feat, edge_index=edge_index, edge_fea=edge_attr, v=vel_0)
            loc_predict = out[0]
        elif model.__class__.__name__ == 'EGHN':
            assert True, 'only for protein dataset'
            local_edge_index, local_edge_attr = MDAnalysisDataset.get_local_edge(config.data_directory)
            local_edge_index, local_edge_attr = local_edge_index.to(device), local_edge_attr.to(device)
            row, col = local_edge_index
            loc_edge_length_0 = torch.sqrt(torch.sum((loc_0[row] - loc_0[col])**2, dim=1)).unsqueeze(1)
            local_edge_attr = torch.cat([local_edge_attr, loc_edge_length_0], dim=1)
            n_node = torch.tensor([loc_0.size(0) // batch_size])
            out = model(x=loc_0, h=node_feat, edge_index=edge_index, edge_fea=edge_attr, local_edge_index=local_edge_index, 
                                local_edge_fea=local_edge_attr, n_node=n_node, v=vel_0)
            loc_predict = out[0]
        elif model.__class__.__name__ == 'GNN':
            nodes = torch.cat([loc_0, vel_0], dim=1)
            loc_predict = model(h=nodes, edge_index=edge_index, edge_fea=edge_attr)
        elif model.__class__.__name__ == 'Linear_dynamics':
            loc_predict = model(x=loc_0, v=vel_0)
        elif model.__class__.__name__ == 'RF_vel':
            vel_norm = torch.sqrt(torch.sum(vel_0 ** 2, dim=1).unsqueeze(1)).detach()
            loc_predict = model(vel_norm=vel_norm, x=loc_0, edges=edge_index, vel=vel_0, edge_attr=edge_attr)
        elif model.__class__.__name__ == 'OurDynamics':  # TFN
            loc_predict = model(loc_0, vel_0, node_attr, edge_index)
        elif model.__class__.__name__ == 'GVPNet':
            h_V = (node_feat, torch.stack([loc_0, vel_0], dim=1))  # node_s, node_v
            row, col = edge_index
            h_E = (edge_attr, (loc_0[row] - loc_0[col]).unsqueeze(1))  # edge_s, edge_v
            out = model(h_V=h_V, edge_index=edge_index, h_E=h_E, batch=data['batch'])
            loc_predict = out[1][:, 0, :]  # get coord
        elif model.__class__.__name__ == 'DimeNet' or model.__class__.__name__ == 'DimeNetPlusPlus':
            loc_predict = model(z=node_feat, pos=loc_0, batch=data['batch'])
        elif model.__class__.__name__ == 'SchNet':
            loc_predict = model(z=node_feat, pos=loc_0, batch=data['batch'], edge_index=edge_index)
        else:
            print(model.__class__.__name__)
            raise Exception('Wrong model')
        
        loss_loc = loss(loc_predict, loc_t)

        # record the loss
        result['loss'] += loss_loc.item() * batch_size
        result['counter'] += batch_size
        

        if model.__class__.__name__ == 'FastEGNN' or model.__class__.__name__ == 'FastRF' \
                or model.__class__.__name__ == 'FastTFN' or model.__class__.__name__ == 'FastSchNet':
            # Add MMD
            node_loc, virtual_node_loc = loc_predict, virtual_node_loc.permute(0, 2, 1)  # [num_node, 3], [B, C, 3]
            C = virtual_node_loc.size(1)
            num_sample = sample * C
            num_sample = min(num_sample, node_loc.size(0))
            if loader.dataset.__class__.__name__ == 'Simulation':
                data_batch = data['batch']

                l_vv, l_rv = 0.0, 0.0
                for i in range(batch_size):
                    # Different node count for each graph
                    node_loc_i = node_loc[data_batch == i, :]  # [batch_node, 3]
                    virtual_node_loc_i = virtual_node_loc[i]  # [C, 3]

                    num_node = node_loc_i.size(0)

                    # Random sample real nodes
                    sample_idx = torch.randperm(num_node)[:num_sample]
                    node_loc_i = node_loc_i[sample_idx]  # [num_sample, 3]

                    # calc_kernel
                    k_vv = kernel(virtual_node_loc_i, virtual_node_loc_i, sigma)
                    k_rv = kernel(node_loc_i, virtual_node_loc_i, sigma)

                    l_vv += torch.sum(k_vv)
                    l_rv += torch.sum(k_rv)
                
                # average between mini-batch
                l_vv = l_vv / batch_size / C / C
                l_rv = 2 * l_rv / batch_size / num_sample / C

            else:
                B, C, _ = virtual_node_loc.size()  # [B, C, 3]
                
                node_loc = node_loc.reshape(B, -1, 3)  # [num_node, 3] => [B, C, 3]
                B, num_node, _ = node_loc.size()

                # Random sample real nodes
                num_sample = min(num_sample, num_node)
                sample_idx = torch.randperm(num_node)[:num_sample]
                node_loc = node_loc[:, sample_idx, :]  # [B, num_sample, 3]

                # calc_kernel
                k_vv = kernel(virtual_node_loc, virtual_node_loc, sigma)  # [B, C, C]
                k_rv = kernel(node_loc, virtual_node_loc, sigma)  # [B, num_sample, C]

                # average between mini-batch
                l_vv = torch.sum(k_vv) / B / C / C
                l_rv = 2 * torch.sum(k_rv) / B / num_sample / C

            loss_mmd = l_vv - l_rv

            loss_loc = loss_loc + weight * loss_mmd

        
        if backprop:
            loss_loc.backward()
            optimizer.step()

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""

    print(f'{prefix + tag} epoch: {epoch_index}, avg loss: {result["loss"] / result["counter"] :.5f}')

    return result['loss'] / result['counter']

def train(model, loader_train, loader_valid, loader_test, optimizer, loss, sigma, weight, log_directory, log_name, early_stop=float('inf'), device='cpu', test_interval=5, sample=3, config=None):
    log_dict = {'epochs': [], 'loss': [], 'loss_train': []}
    best_log_dict = {'epoch_index': 0, 'loss_valid': 1e8, 'loss_test': 1e8, 'loss_train': 1e8}

    start =time.perf_counter()
    for epoch_index in range(1, int(1e6)):
        loss_train = train_single_epoch(model, loader_train, optimizer, loss, sigma, weight, epoch_index, backprop=True, tag='train', device=device, sample=sample, config=config)
        log_dict['loss_train'].append(loss_train)

        if epoch_index % test_interval == 0:
            loss_valid = train_single_epoch(model, loader_valid, optimizer, loss, sigma, weight, epoch_index, backprop=False, tag='valid', device=device, sample=sample, config=config)
            loss_test = train_single_epoch(model, loader_test, optimizer, loss, sigma, weight, epoch_index, backprop=False, tag='test', device=device, sample=sample, config=config)
            
            log_dict['epochs'].append(epoch_index)
            log_dict['loss'].append(loss_test)
            
            if loss_valid < best_log_dict['loss_valid']:
                best_log_dict = {'epoch_index': epoch_index, 'loss_valid': loss_valid, 'loss_test': loss_test, 'loss_train': loss_train}
                name = None
                if config.dataset_name == '100_0_0':
                    name = 'nbody'
                elif config.dataset_name == 'adk':
                    name = 'protein'
                elif config.dataset_name == 'Water-3D':
                    name = 'Water-3D'

                os.makedirs(f'./state_dict/{name}', exist_ok=True)
                torch.save(model.state_dict(), f'./state_dict/{name}/{model.__class__.__name__}_best_model.pth')
            print(f'*** Best Valid Loss: {best_log_dict["loss_valid"] :.5f} | Best Test Loss: {best_log_dict["loss_test"] :.5f} | Best Epoch Index: {best_log_dict["epoch_index"]}')

            if epoch_index - best_log_dict['epoch_index'] >= early_stop:
                best_log_dict['early_stop'] = epoch_index
                print(f'Early stopped! Epoch: {epoch_index}')
                break

        end = time.perf_counter() 
        time_cost = end - start
        best_log_dict['time_cost'] = time_cost
        
        json_object = json.dumps([best_log_dict, log_dict], indent=4)
        os.makedirs(log_directory, exist_ok=True)
        with open(f'{log_directory}/{log_name}', "w") as outfile:
            outfile.write(json_object)
    

    return best_log_dict, log_dict
