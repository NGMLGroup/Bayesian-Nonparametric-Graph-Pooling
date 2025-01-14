import os
import os.path as osp
import pytorch_lightning as pl

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import HeterophilousGraphDataset, WikipediaNetwork, WebKB
from torch_geometric.transforms import Constant, Compose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import homophily
from ogb.graphproppred import PygGraphPropPredDataset

# Local imports
from source.utils import LabelsToInt, get_train_val_test_datasets
from .torch_datasets import (GraphClassificationBench, 
                             PyGSPDataset, 
                             TUDataset)


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
        if self.train_dataset[0].edge_attr is not None:
            self.num_edge_features = self.train_dataset[0].edge_attr.size(1)
        else:
            self.num_edge_features = None
        self.num_features = orig_train_dataset.num_features
        self.num_classes = orig_train_dataset.num_classes
        self.avg_nodes = int(orig_train_dataset.data.num_nodes / len(orig_train_dataset))
        self.max_nodes = max([d.num_nodes for d in orig_train_dataset])

        self.seed = args.seed
        self.n_folds = args.n_folds
        if args.fold_id is not None:
            assert args.fold_id < args.n_folds
            self.fold_id = args.fold_id
        else:
            self.fold_id = 0

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
        if self.dataset[0].edge_attr is not None:
            self.num_edge_features = self.dataset[0].edge_attr.size(1)
        else:
            self.num_edge_features = None
        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes
        self.avg_nodes = int(self.dataset.data.num_nodes / len(self.dataset)) # why like this and not with sum()/len()?
        self.max_nodes = max([d.num_nodes for d in self.dataset])

        self.train_dataset, self.val_dataset, self.test_dataset = get_train_val_test_datasets(self.dataset, args.seed, args.n_folds, args.fold_id)
        
        self.seed = args.seed
        self.n_folds = args.n_folds
        if args.fold_id is not None:
            assert args.fold_id < args.n_folds
            self.fold_id = args.fold_id
        else:
            self.fold_id = 0
    
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

        if args.get('reload', False):    
            # Delete pre-processed data
            processed_path = osp.join(path, '_'.join(name.split('-')), 'processed')
            if osp.exists(processed_path):
                print(f"Force reload: deleting {processed_path}")
                os.system(f"rm -r {processed_path}")

        if name in ['ogbg-ppa', 'ogbg-molhiv']:
            self.dataset = PygGraphPropPredDataset(name=name, root=path, pre_transform=transforms)
        else:
            raise NotImplementedError(f"Dataset {name} not supported")
        if self.dataset[0].edge_attr is not None:
            self.num_edge_features = self.dataset[0].edge_attr.size(1)
        else:
            self.num_edge_features = None
        split_idx = self.dataset.get_idx_split()
        
        orig_train_dataset = self.dataset[split_idx["train"]]
        orig_val_dataset = self.dataset[split_idx["valid"]]
        orig_test_dataset = self.dataset[split_idx["test"]]

        train_data_list = [data for data in orig_train_dataset]
        val_data_list = [data for data in orig_val_dataset]
        test_data_list = [data for data in orig_test_dataset]

        for data in train_data_list:
            data.y = data.y.squeeze()

        for data in val_data_list:
            data.y = data.y.squeeze()

        for data in test_data_list:
            data.y = data.y.squeeze()

        self.train_dataset, self.val_dataset, self.test_dataset = train_data_list, val_data_list, test_data_list

        self.num_features = orig_train_dataset.num_features
        self.num_classes = orig_train_dataset.num_classes
        self.avg_nodes = int(orig_train_dataset.data.num_nodes / len(orig_train_dataset))
        self.max_nodes = max([d.num_nodes for d in orig_train_dataset])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.args.batch_size, shuffle=True, 
                          num_workers=self.args.get('num_workers', 0))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.args.batch_size, 
                          num_workers=self.args.get('num_workers', 0))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.args.batch_size,
                          num_workers=self.args.get('num_workers', 0))


class PyGSPDataModule(pl.LightningDataModule):
    def __init__(self, 
                 args,
                 pre_transform=None):
        super().__init__()
        self.args = args
        path = "data/PyGSP"

        self.dataset = PyGSPDataset(root=path, name=args.pygsp_graph, 
                                    pre_transform=pre_transform, force_reload=args.reload)
        self.num_features = self.dataset.num_features

    def train_dataloader(self):
        return DataLoader(self.dataset, self.args.batch_size)