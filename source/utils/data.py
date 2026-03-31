import random
from sklearn.model_selection import KFold
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import sort_edge_index


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


class SortNodes(BaseTransform):
    """
    Sort the nodes of the graph according to the node label.
    """

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
            data.edge_index, data.edge_weight = sort_edge_index(edge_index_renamed, data.edge_weight)

        return data
