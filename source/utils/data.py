import random
from typing import Optional
from sklearn.model_selection import KFold
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (add_self_loops, 
                                   coalesce, 
                                   get_laplacian,
                                   sort_edge_index)


class LabelsToInt(BaseTransform):
    def __call__(self, data):
        data.y = data.y.to(torch.long)
        return data
    

def get_train_val_test_datasets(data, seed, n_folds, fold_id, ratio=0.85):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_splits = [k for k in kf.split(data)]
    trainval_indices, test_indices = all_splits[fold_id]
    trainval_indices, test_indices = trainval_indices.tolist(), test_indices.tolist()
    trainval_dataset = [data[i] for i in trainval_indices]
    test_dataset = [data[i] for i in test_indices]

    indices = list(range(len(trainval_dataset)))
    random.seed(seed)
    random.shuffle(indices)
    split_index = int(ratio * len(indices))
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    train_dataset = [trainval_dataset[i] for i in train_indices]
    val_dataset = [trainval_dataset[i] for i in val_indices]

    return train_dataset, val_dataset, test_dataset


class NormalizeAdjSparse_with_ea(BaseTransform):
    r"""
    Applies the following transformation:
    :math:`A \rightarrow I - \delta \cdot L`.
    using sparse matrix representations.
    
    This transformation also accounts for edge attributes.
    Edge attributes of added self-loops are set to zero-vector.
    """

    def __init__(self, delta: float = 0.85) -> None:
        self.delta = delta
        super().__init__()

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        N = data.num_nodes

        edge_index, edge_weight = data.edge_index, data.edge_weight

        # Check how many edges have self loops
        self_loop_mask = edge_index[0] == edge_index[1]
        initial_self_loops = self_loop_mask.sum().item()

        # Get the symmetrically normalized Laplacian (I - D^-.5 A D^-.5) in sparse format
        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization='sym', num_nodes=N
        )

        # Check if new self loops have been added
        new_self_loop_mask = edge_index[0] == edge_index[1]
        num_new_self_loops = new_self_loop_mask.sum().item() - initial_self_loops

        # Rescale the Laplacian weights by -delta
        edge_weight = -self.delta * edge_weight

        # Add self-loops representing the identity matrix
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=N
        )

        # Prepare edge attributes for coalescing
        if data.edge_attr is not None:

            if num_new_self_loops > 0:
                num_self_loops = 2*N # self loops from Laplacian and self loops from add_self_loops
            else:
                num_self_loops = N # self loops only from add_self_loops

            # Create zero edge attributes for the self-loops
            attr_dim = data.edge_attr.size(1)
            self_loop_attr = torch.zeros(
                num_self_loops, attr_dim, device=data.edge_attr.device
            )

            # Concatenate original edge attributes and self-loop attributes
            edge_attr = torch.cat([data.edge_attr, self_loop_attr], dim=0)

        else:
            edge_attr = None

        # Prepare edge values for coalescing
        if edge_attr is not None:
            edge_weight = edge_weight.unsqueeze(1)  # Shape: [num_edges + num_self_loops, 1]
            edge_value = torch.cat([edge_weight, edge_attr], dim=1)  # Shape: [num_edges + num_self_loops, 1 + attr_dim]
        else:
            edge_value = edge_weight  # Shape: [num_edges + num_self_loops]

        # Coalesce the sparse matrix to remove duplicate entries and sum their values
        edge_index, edge_value = coalesce(edge_index, edge_value, N)

        # Split edge_value back into edge_weight and edge_attr
        if edge_attr is not None:
            edge_weight = edge_value[:, 0]
            edge_attr = edge_value[:, 1:].to(data.edge_attr.dtype)
            data.edge_attr = edge_attr
        else:
            edge_weight = edge_value

        data.edge_index = edge_index
        data.edge_weight = edge_weight

        return data


class SortNodes(BaseTransform):
    """
    Sort the nodes of the graph according to the node label.
    """

    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        assert data.y is not None
        y_sorted, sort_idx = torch.sort(data.y)
        edge_index_renamed = torch.empty_like(data.edge_index)
        for new_i in range(data.num_nodes):
            i = sort_idx[new_i]
            mask_i = data.edge_index == i
            edge_index_renamed[mask_i] = new_i

        data.x = data.x[sort_idx]
        data.y = y_sorted
        
        # sort edge_index_renamed in order to have edges ordered by source
        if data.edge_attr is not None:
            data.edge_index, (data.edge_weight, data.edge_attr) = sort_edge_index(edge_index_renamed,
                                                                                  edge_attr=[data.edge_weight,
                                                                                             data.edge_attr])
        else:
            data.edge_index, data.edge_weight = sort_edge_index(edge_index_renamed,data.edge_weight)

        return data


def cluster_to_s(cluster_index: Tensor, 
                 node_index: Optional[Tensor] = None,
                 weight: Optional[Tensor] = None,
                 as_edge_index: bool = False,
                 num_nodes: Optional[int] = None,
                 num_clusters: Optional[int] = None):
    if num_nodes is None:
        num_nodes = cluster_index.size(0)
    if node_index is None:
        node_index = torch.arange(num_nodes, dtype=torch.long,
                                  device=cluster_index.device)
    if as_edge_index:
        return torch.stack([node_index, cluster_index], dim=0), weight
    else:
        return SparseTensor(row=node_index, col=cluster_index, value=weight,
                            sparse_sizes=(num_nodes, num_clusters))