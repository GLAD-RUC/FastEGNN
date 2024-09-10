import os
import json
import argparse

import torch
from torch import nn, optim

from datasets.qm9 import dataset
from datasets.qm9.models import EGNN
from datasets.qm9 import utils as qm9_utils

from datasets.qm9.FastEGNNQM9 import FastEGNNQM9


parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='./logs/qm9', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=128, metavar='N',
                    help='learning rate')
parser.add_argument('--attention', type=int, default=1, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=7, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--property', type=str, default='homo', metavar='N',
                    help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                    help='maximum power to take into one-hot features')
parser.add_argument('--dataset_paper', type=str, default="cormorant", metavar='N',
                    help='cormorant, lie_conv')
parser.add_argument('--node_attr', type=int, default=0, metavar='N',
                    help='node_attr or not')
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='weight decay')
parser.add_argument('--data_dir', type=str, required=True)

parser.add_argument('--device', type=str, default='cpu', help='device (default: cpu)')
parser.add_argument('--cutoff_rate', type=float, default=0.25, help='cutoff rate of edge_rr (default: 0.25)')
parser.add_argument('--virtual_channel', type=int, required=True, help='channel count of virtual node')


args = parser.parse_args()
device = args.device
dtype = torch.float32
print(args)

# block multi-experiment
os.makedirs(args.outf, exist_ok=True)
os.makedirs(args.outf + "/" + args.exp_name, exist_ok=False)


dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.data_dir, args.num_workers, args.cutoff_rate)
# compute mean and mean absolute deviation
meann, mad = qm9_utils.compute_mean_mad(dataloaders, args.property)


model = FastEGNNQM9(node_feat_nf=15, node_attr_nf=0, edge_attr_nf=1, hidden_nf=args.nf, virtual_channels=args.virtual_channel, 
                    device=args.device, n_layers=args.n_layers, residual=True, attention=args.attention)

print(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l1 = nn.L1Loss()


def train(epoch, loader, partition='train'):
    lr_scheduler.step()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()

        data = data.to(device)
        batch_size = data['batch'][-1].item() + 1
        node_loc = data['positions'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)


        node_feat = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)
        edge_index = data['edge_index'].to(device)
        label = data[args.property].to(device, dtype)
        edge_attr = data['edge_attr'].to(device, dtype)

        pred = model(node_feat=node_feat, node_loc=node_loc, edge_index=edge_index, data_batch=data['batch'], 
                     loc_mean=data['loc_mean'].to(device, dtype), edge_attr=edge_attr)


        if partition == 'train':
            loss = loss_l1(pred, (label - meann) / mad)
            loss.backward()
            optimizer.step()
        else:
            loss = loss_l1(mad * pred + meann, label)

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % args.log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
    return res['loss'] / res['counter']


if __name__ == "__main__":
    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    for epoch in range(0, args.epochs):
        train_loss = train(epoch, dataloaders['train'], partition='train')
        if epoch % args.test_interval == 0:
            val_loss = train(epoch, dataloaders['valid'], partition='valid')
            test_loss = train(epoch, dataloaders['test'], partition='test')
            res['epochs'].append(epoch)
            res['losess'].append(test_loss)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
            print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
            print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))


            json_object = json.dumps(res, indent=4)
            with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
                outfile.write(json_object)
