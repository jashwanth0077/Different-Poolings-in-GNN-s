import os
import pickle
import random
import numpy as np
import torch
from typing import Any
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import InMemoryDataset


def log(**kwargs):
    def _map(value: Any) -> str:
        if isinstance(value, int) and not isinstance(value, bool):
            return f'{value:03d}'
        if isinstance(value, float):
            return f'{value:.4f}'
        return value

    print(', '.join(f'{key}: {_map(value)}' for key, value in kwargs.items()))
  
    
class DataToFloat(BaseTransform):
    def __call__(self, data):
        data.x = data.x.to(torch.float32)
        return data
    

class EXPWL1Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(EXPWL1Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["EXPWL1.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/EXPWL1.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    

def negative_edges(edge_index, num_nodes=None, num_neg_samples=None,
                      force_undirected=False):
    r"""Taken from https://github.com/AaltoPML/Rethinking-pooling-in-GNNs
    
    Samples random negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        num_neg_samples (int, optional): The number of negative samples to
            return. If set to :obj:`None`, will try to return a negative edge
            for every positive edge. (default: :obj:`None`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    num_neg_samples = num_neg_samples or edge_index.size(1)

    # Handle '|V|^2 - |E| < |E|' case for G = (V, E).
    num_neg_samples = num_nodes * num_nodes - edge_index.size(1)

    if force_undirected:
        num_neg_samples = num_neg_samples // 2

        # Upper triangle indices: N + ... + 1 = N (N + 1) / 2
        rng = range((num_nodes * (num_nodes + 1)) // 2)

        # Remove edges in the lower triangle matrix.
        row, col = edge_index
        mask = row <= col
        row, col = row[mask], col[mask]

        # idx = N * i + j - i * (i+1) / 2
        idx = (row * num_nodes + col - row * (row + 1) // 2).to('cpu')
    else:
        rng = range(num_nodes**2)
        # idx = N * i + j
        idx = (edge_index[0] * num_nodes + edge_index[1]).to('cpu')

    perm = torch.tensor(rng)
    # mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
    if hasattr(torch, 'isin'):
        mask = torch.isin(perm, idx)
    else:
        mask = (perm.unsqueeze(1) == idx).any(dim=1)
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        # mask = torch.from_numpy(np.isin(tmp, idx)).to(torch.bool)
        if hasattr(torch, 'isin'):
            mask = torch.isin(tmp, idx)
        else:
            mask = (perm.unsqueeze(1) == idx).any(dim=1)
        perm[rest] = tmp
        rest = rest[mask.nonzero().view(-1)]

    if force_undirected:
        # (-sqrt((2 * N + 1)^2 - 8 * perm) + 2 * N + 1) / 2
        row = torch.floor((-torch.sqrt((2. * num_nodes + 1.)**2 - 8. * perm) +
                           2 * num_nodes + 1) / 2)
        col = perm - row * (2 * num_nodes - row - 1) // 2
        neg_edge_index = torch.stack([row, col], dim=0).long()
        neg_edge_index = to_undirected(neg_edge_index)
    else:
        row = perm / num_nodes
        col = perm % num_nodes
        neg_edge_index = torch.stack([row, col], dim=0).long()

    return neg_edge_index.to(edge_index.device)


def batched_negative_edges(edge_index, batch, num_neg_samples=None,
                              force_undirected=False):
    r"""Taken from https://github.com/AaltoPML/Rethinking-pooling-in-GNNs
    
    Samples random negative edges of multiple graphs given by
    :attr:`edge_index` and :attr:`batch`.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        num_neg_samples (int, optional): The number of negative samples to
            return. If set to :obj:`None`, will try to return a negative edge
            for every positive edge. (default: :obj:`None`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor
    """
    split = degree(batch[edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(edge_index, split, dim=1)
    num_nodes = degree(batch, dtype=torch.long)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])

    neg_edge_indices = []
    for edge_index, N, C in zip(edge_indices, num_nodes.tolist(),
                                cum_nodes.tolist()):
        neg_edge_index = negative_edges(edge_index - C, N, num_neg_samples,
                                           force_undirected) + C
        neg_edge_indices.append(neg_edge_index)

    return torch.cat(neg_edge_indices, dim=1)