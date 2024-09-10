import torch
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from datasets.qm9.data.utils import initialize_datasets
from datasets.qm9.args import init_argparse
from datasets.qm9.data.collate import collate_fn
from utils.rotate import random_rotate



class PygQM9Dataset(Dataset):
    def __init__(self, dataset, partition='train', cutoff_rate=0):
        super(Dataset, self).__init__()
        self.partition = partition
        self.cutoff_rate=cutoff_rate
        self.data = []
        self.trans_to_pyg(dataset)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def trans_to_pyg(self, dataset):
        for d in tqdm(dataset, total=len(dataset)):
            num_atoms = d['num_atoms'].item()
            pos = d['positions'][:num_atoms]

            # add rotation in test dataset
            rotate_matrix = random_rotate()
            rotate_matrix = rotate_matrix.to(pos.device).to(pos.dtype)

            if self.partition == 'test':
                pos = pos @ rotate_matrix

            # calc edge_index
            edge_index = self.cutoff_edge(pos)
            row, col = edge_index
            edge_length = torch.sqrt(torch.sum((pos[row] - pos[col])**2, dim=1)).unsqueeze(1)

            loc_mean = torch.mean(pos, dim=0).unsqueeze(-1).unsqueeze(0)  # [1, 3, 1]

            self.data.append(
                Data(charges=d['charges'][:num_atoms], positions=pos, index=d['index'], A=d['A'], edge_index=edge_index,
                     B=d['B'], C=d['C'], mu=d['mu'], alpha=d['alpha'], homo=d['homo'], lumo=d['lumo'], gap=d['gap'],
                     r2=d['r2'], zpve=d['zpve'], U0=d['U0'], U=d['U'], H=d['H'], G=d['G'], Cv=d['Cv'], omega1=d['omega1'],
                     zpve_thermo=d['zpve_thermo'], U0_thermo=d['U0_thermo'], U_thermo=d['U_thermo'], H_thermo=d['H_thermo'],
                     G_thermo=d['G_thermo'], Cv_thermo=d['Cv_thermo'], one_hot=d['one_hot'][:num_atoms], edge_attr=edge_length,
                     loc_mean=loc_mean,
                     )
            )

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


def retrieve_dataloaders(batch_size, datadir, num_workers=1, cutoff_rate=0):
    # Initialize dataloader
    args = init_argparse('qm9')
    args, datasets, num_species, charge_scale = initialize_datasets(args, datadir, 'qm9',
                                                                    subtract_thermo=args.subtract_thermo,
                                                                    force_download=False
                                                                    )
    qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                 'lumo': 27.2114}

    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)

    # Convert to PygDataloader
    pyg_datasets = {}
    for k, v in datasets.items():
        pyg_datasets[k] = PygQM9Dataset(v, partition=k, cutoff_rate=cutoff_rate)

    # Construct PyTorch Geometric dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=num_workers,)
                         for split, dataset in pyg_datasets.items()}

    return dataloaders, charge_scale



def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


if __name__ == '__main__':
    '''
    dataloader = retrieve_dataloaders(batch_size=64)
    for i, batch in enumerate(dataloader['train']):
        print(i)
    '''
