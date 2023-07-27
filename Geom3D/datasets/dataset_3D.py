
import os
import pandas as pd
import numpy as np
from itertools import repeat
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import subgraph, to_networkx, remove_self_loops, to_dense_adj, dense_to_sparse
from torch_sparse import coalesce, spspmm


def extend_graph(data):
    edge_index = data.edge_index
    N = data.num_nodes

    value = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)

    index, value = spspmm(edge_index, value, edge_index, value, N, N, N)
    value.fill_(0)
    index, value = remove_self_loops(index, value)

    edge_index = torch.cat([edge_index, index], dim=1)

    edge_index, _ = coalesce(edge_index, None, N, N)

    value = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)

    index, value = spspmm(edge_index, value, edge_index, value, N, N, N)
    value.fill_(0)
    index, value = remove_self_loops(index, value)

    edge_index = torch.cat([edge_index, index], dim=1)

    data.extended_edge_index, _ = coalesce(edge_index, None, N, N)
    return data


class Molecule3DDataset(InMemoryDataset):
    def __init__(self, root, dataset, mask_ratio=0, remove_center=False, transform=None, pre_transform=None, pre_filter=None, empty=False, use_extend_graph=False):
        self.root = root
        self.dataset = dataset
        self.mask_ratio = mask_ratio
        self.remove_center = remove_center
        self.use_extend_graph = use_extend_graph

        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super(Molecule3DDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Dataset: {}\nData: {}'.format(self.dataset, self.data))

    def subgraph(self, data):
        G = to_networkx(data)
        node_num = data.x.size()[0]
        sub_num = int(node_num * (1 - self.mask_ratio))

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

        # BFS
        while len(idx_sub) <= sub_num:
            if len(idx_neigh) == 0:
                idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
                idx_neigh = set([np.random.choice(idx_unsub)])
            sample_node = np.random.choice(list(idx_neigh))

            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(
                set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

        idx_nondrop = idx_sub
        idx_nondrop.sort()

        edge_idx, edge_attr = subgraph(
            subset=idx_nondrop,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            relabel_nodes=True,
            num_nodes=node_num
        )
        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
        data.positions = data.positions[idx_nondrop]
        data.__num_nodes__ = data.x.size()[0]
        
        if "radius_edge_index" in data:
            radius_edge_index, _ = subgraph(
                subset=idx_nondrop,
                edge_index=data.radius_edge_index,
                relabel_nodes=True, 
                num_nodes=node_num)
            data.radius_edge_index = radius_edge_index
        if "extended_edge_index" in data:
            # TODO: may consider extended_edge_attr
            extended_edge_index, _ = subgraph(
                subset=idx_nondrop,
                edge_index=data.extended_edge_index,
                relabel_nodes=True, 
                num_nodes=node_num)
            data.extended_edge_index = extended_edge_index
        # TODO: will also need to do this for other edge_index
        return data

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        if self.use_extend_graph:
            data = extend_graph(data)

        if self.mask_ratio > 0:
            data = self.subgraph(data)

        if self.remove_center:
            center = data.positions.mean(dim=0)
            data.positions -= center

        return data
    
    def _download(self):
        return

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        return


if __name__ == "__main__":

    def extend_graph(data):
        edge_index = data.edge_index
        N = data.num_nodes
        
        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)
        edge_index_2_hop, value_2_hop = spspmm(edge_index, value, edge_index, value, N, N, N)
        print("edge_index_2_hop", edge_index_2_hop)
        print("value_2_hop", value_2_hop)
        value_2_hop.fill_(1)
        edge_index_3_hop, value_3_hop = spspmm(edge_index, value, edge_index_2_hop, value_2_hop, N, N, N)
        print("edge_index_3_hop", edge_index_3_hop)
        print("value_3_hop", value_3_hop)
        value_3_hop.fill_(1)
        
        index_list = [edge_index, edge_index_2_hop, edge_index_3_hop]
        value_list = [value, value_2_hop, value_3_hop]
        index = torch.cat(index_list, dim=-1)
        value = torch.cat(value_list, dim=-1)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        
        data.extended_edge_index, _ = coalesce(edge_index, None, N, N)
        return data

    from torch import Tensor
    x = Tensor([0, 1, 2, 3, 4])
    row = Tensor([0, 1, 1, 2, 2, 3, 3, 4])
    col = Tensor([1, 0, 2, 1, 3, 2, 4, 3])
    edge_index = [row, col]
    edge_index = torch.stack(edge_index).long()
    data = Data(
        x=x,
        edge_index=edge_index,
    )
    print(data)

    data = extend_graph(data)
    print()
    print(data.extended_edge_index)
    print(data)