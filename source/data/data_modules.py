import os.path as osp
import shutil
import pytorch_lightning as pl

from torch_geometric.datasets import LRGBDataset
from torch_geometric.transforms import Constant, Compose
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset

# Local imports
from source.utils import (
    LabelsToInt,
    get_train_val_test_datasets,
)
from .torch_datasets import (GraphClassificationBench, 
                             MultipartiteGraphDataset,
                             TUDataset)


def _get_num_edge_features(data):
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is None:
        return None
    if edge_attr.dim() == 1:
        return 1
    if edge_attr.size(-1) == 0:
        return None
    return edge_attr.size(-1)


def _infer_num_classes(dataset, override=None):
    if override is not None:
        return override

    num_classes = getattr(dataset, "num_classes", None)
    if num_classes is not None:
        return num_classes

    y = dataset[0].y
    if y.dim() == 0:
        return 1
    return y.numel()


def _set_graph_dataset_stats(module, dataset, num_classes=None):
    storage = getattr(dataset, "_data", None)
    if storage is None:
        storage = getattr(dataset, "data", None)

    module.num_edge_features = _get_num_edge_features(dataset[0]) or 0
    module.num_features = dataset.num_features
    module.num_classes = _infer_num_classes(dataset, override=num_classes)

    if storage is not None and getattr(storage, "num_nodes", None) is not None:
        total_nodes = storage.num_nodes
    else:
        total_nodes = sum(data.num_nodes for data in dataset)

    module.avg_nodes = int(total_nodes / len(dataset))
    module.max_nodes = max(data.num_nodes for data in dataset)


def _squeeze_targets(dataset):
    for data in dataset:
        data.y = data.y.squeeze()


class BenchHardDataModule(pl.LightningDataModule):
    def __init__(self, 
                 args, 
                 pre_transform=None,
                 force_reload=True):
        super().__init__()
        self.args = args
        path = "data/Bench-hard/"
        orig_train_dataset = GraphClassificationBench(path, split='train', easy=False, 
                                                      small=False, pre_transform=pre_transform,
                                                      force_reload=force_reload)
        orig_val_dataset = GraphClassificationBench(path, split='val', easy=False, 
                                                    small=False, pre_transform=pre_transform,
                                                    force_reload=force_reload)
        orig_test_dataset = GraphClassificationBench(path, split='test', easy=False, 
                                                     small=False, pre_transform=pre_transform,
                                                     force_reload=force_reload)
        
        train_data_list = [data for data in orig_train_dataset]
        val_data_list = [data for data in orig_val_dataset]
        test_data_list = [data for data in orig_test_dataset]
        data_list = train_data_list + val_data_list + test_data_list
        self.train_dataset, self.val_dataset, self.test_dataset = get_train_val_test_datasets(data_list, args.seed, args.n_folds, args.fold_id)
        _set_graph_dataset_stats(self, orig_train_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.args.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.args.batch_size)
    

class MultipartiteDataModule(pl.LightningDataModule):
    def __init__(self,
                 args,
                 pre_transform=None,
                 force_reload=True):
        super().__init__()
        self.args = args
        self.dataset = MultipartiteGraphDataset(
            root="data/Multipartite",
            pre_transform=pre_transform,
            force_reload=force_reload,
        )
        _set_graph_dataset_stats(self, self.dataset)
        self.train_dataset, self.val_dataset, self.test_dataset = get_train_val_test_datasets(
            self.dataset, args.seed, args.n_folds, args.fold_id
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.args.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.args.batch_size)


class TUDataModule(pl.LightningDataModule):
    def __init__(self, 
                 name,
                 args, 
                 pre_transform=None,
                 force_reload=True):
        super().__init__()
        self.args = args

        custom_trans = args.get('transform', [])
        new_trans = []
        if 'constant' in custom_trans:
            new_trans.append(Constant())
        if 'labels_to_int' in custom_trans:
            new_trans.append(LabelsToInt())
        transforms = Compose(new_trans) if pre_transform is None else Compose([pre_transform, *new_trans])
        
        self.dataset = TUDataset(
            root="data/TUDataset", 
            name=name, 
            cleaned=args.get('clean', True), 
            use_node_attr=args.get('use_node_attr', False),
            pre_transform=transforms, 
            force_reload=force_reload)
        self.dataset = self.dataset.shuffle()
        _set_graph_dataset_stats(self, self.dataset)
        self.train_dataset, self.val_dataset, self.test_dataset = get_train_val_test_datasets(self.dataset, args.seed, args.n_folds, args.fold_id)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.args.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.args.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.args.batch_size)
    

class OGBDataModule(pl.LightningDataModule):
    def __init__(self, 
                 name,
                 args, 
                 pre_transform=None):
        super().__init__()
        self.args = args
        path = "data/ogb/"

        custom_trans = args.get('transform', [])
        new_trans = []
        if 'constant' in custom_trans:
            new_trans.append(Constant())
        transforms = Compose(new_trans) if pre_transform is None else Compose([pre_transform, *new_trans])

        processed_path = osp.join(path, '_'.join(name.split('-')), 'processed')
        if osp.exists(processed_path):
            print(f"Deleting cached processed data at {processed_path}")
            shutil.rmtree(processed_path)

        if name in ['ogbg-ppa', 'ogbg-molhiv']:
            self.dataset = PygGraphPropPredDataset(name=name, root=path, pre_transform=transforms)
        else:
            raise NotImplementedError(f"Dataset {name} not supported")
        split_idx = self.dataset.get_idx_split()
        
        orig_train_dataset = self.dataset[split_idx["train"]]
        orig_val_dataset = self.dataset[split_idx["valid"]]
        orig_test_dataset = self.dataset[split_idx["test"]]

        train_data_list = [data for data in orig_train_dataset]
        val_data_list = [data for data in orig_val_dataset]
        test_data_list = [data for data in orig_test_dataset]

        _squeeze_targets(train_data_list)
        _squeeze_targets(val_data_list)
        _squeeze_targets(test_data_list)

        self.train_dataset, self.val_dataset, self.test_dataset = train_data_list, val_data_list, test_data_list

        _set_graph_dataset_stats(self, orig_train_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.args.batch_size, shuffle=True, 
                          num_workers=self.args.get('num_workers', 0))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.args.batch_size, 
                          num_workers=self.args.get('num_workers', 0))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.args.batch_size,
                          num_workers=self.args.get('num_workers', 0))


class LRGBDataModule(pl.LightningDataModule):
    def __init__(self,
                 name,
                 args,
                 pre_transform=None):
        super().__init__()
        self.args = args
        path = "data/lrgb/"

        self.train_dataset = LRGBDataset(
            root=path,
            name=name,
            split='train',
            pre_transform=pre_transform,
            force_reload=True,
        )
        self.val_dataset = LRGBDataset(
            root=path,
            name=name,
            split='val',
            pre_transform=pre_transform,
            force_reload=True,
        )
        self.test_dataset = LRGBDataset(
            root=path,
            name=name,
            split='test',
            pre_transform=pre_transform,
            force_reload=True,
        )

        _set_graph_dataset_stats(self, self.train_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.args.batch_size, shuffle=True,
                          num_workers=self.args.get('num_workers', 0))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.args.batch_size,
                          num_workers=self.args.get('num_workers', 0))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.args.batch_size,
                          num_workers=self.args.get('num_workers', 0))
