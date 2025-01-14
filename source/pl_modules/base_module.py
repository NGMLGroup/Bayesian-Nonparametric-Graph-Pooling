from typing import Optional, Mapping, Type
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torch_geometric.utils import to_scipy_sparse_matrix
from einops import einsum
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch_sparse import SparseTensor

class BaseModule(pl.LightningModule):
    """
    🧱 Base Lightning Module class
    """

    def maybe_log_stuff(self, 
        batch, 
        batch_idx, 
        pooled_adj, 
        pooled_batch, 
        s, 
        plot_type=[], 
        istest=False):
        """
        📊 Log plots at a specific epoch
        """

        if self.plot_preds_at_epoch is not None:
            b_idx = self.plot_preds_at_epoch.get('batch', 0)
            s_idx = self.plot_preds_at_epoch.get('samples', 1)
            every = self.plot_preds_at_epoch.get('every', 1)            

            if batch_idx == b_idx and (self.current_epoch%every==0 or istest):
                mask = batch.batch[batch.edge_index[0]] == s_idx
                sample_edge_index = batch.edge_index[:, mask]
                sample_edge_index = sample_edge_index - batch.ptr[s_idx]
                sample_num_nodes = batch.ptr[s_idx+1] - batch.ptr[s_idx]
                sample_indices = batch.ptr[s_idx] + torch.arange(sample_num_nodes).to(batch.ptr.device)

                if batch.x.shape[-1] == 2:
                    pos = batch[s_idx].x.cpu().detach().numpy()
                else:
                    pos=None

                if s is not None:
                    if isinstance(s, SparseTensor):
                        s = s.to_dense()
                        sample_pool_indices = pooled_batch == s_idx
                        s = s[sample_indices.cpu().detach().numpy()][:, sample_pool_indices.cpu().detach().numpy()]
                    else:
                        s = s[s_idx,:sample_num_nodes.item()]
                    s = s.cpu().detach().numpy()
                
                if 'pooled_graph' in plot_type:
                    title = 'pooled_graph'
                    if istest:
                        title += '_test'
                    if isinstance(pooled_adj, torch.Tensor) and pooled_adj.is_sparse:
                        pooled_adj = to_scipy_sparse_matrix(pooled_adj)
                    else:
                        pooled_adj = pooled_adj.cpu().detach().numpy()
                        pooled_adj = pooled_adj[s_idx]
                    if pos is not None:
                        pos_ = einsum(pos, s[s_idx], 'n f, n k -> k f')
                    else:
                        pos_ = None
                    self.logger.log_nx_graph_plot(
                        pooled_adj,  
                        pos=pos_,
                        name=title,
                        global_step=self.global_step)

                if 'assignments' in plot_type:
                    title = f'assignments'
                    if istest:
                        title += '_test'
                    sample_edge_index = sample_edge_index[:, sample_edge_index[0] != sample_edge_index[1]] # remove self loops
                    self.logger.log_nx_graph_plot(
                        to_scipy_sparse_matrix(sample_edge_index, num_nodes=sample_num_nodes),
                        pos=pos,
                        signal=s.argmax(axis=-1),
                        node_size=25,
                        font_size=8,
                        cmap='tab20',
                        name=title,
                        global_step=self.global_step)

                if 'matrix_assignments' in plot_type:
                    title = f'matrix_assignments'
                    if istest:
                        title += '_test'
                    f, ax = plt.subplots(figsize=(4, 5))
                    ax = sns.heatmap(s, cbar=True, ax=ax)
                    ax.set_xlabel('Clusters')
                    ax.set_ylabel('Nodes')
                    ax.set_title(r'$\mathbf{S}$')
                    self.logger.log_figure(f, title, global_step=self.global_step)
                    plt.close(f)

                if 'rec_adj' in plot_type:
                    title = f'rec_adj'
                    if istest:
                        title += '_test'         
                    if self.model.pooler in ['baypool']:
                        K = self.model.pool.K.detach().cpu().numpy()
                        rec_adj = s @ K @ s.T
                        rec_adj = 1 / (1 + np.exp(-rec_adj))
                    else:
                        rec_adj = s @ s.T
                    f, ax = plt.subplots()
                    ax = sns.heatmap(rec_adj)
                    ax.set_title(r'$\mathbf{S}\mathbf{S}^\top$')
                    self.logger.log_figure(f, title, global_step=self.global_step)
                    plt.close(f)

                if 'gen_matrix' in plot_type:
                    if self.model.pooler in ['baypool']:
                        title = f'gen_matrix'
                        if istest:
                            title += '_test'
                        f, ax = plt.subplots(figsize=(6.5, 5))
                        K = self.model.pool.K.detach().cpu().numpy()
                        ax = sns.heatmap(K, cbar=True, ax=ax)
                        ax.set_title(r'$\mathbf{K}$')
                        self.logger.log_figure(f, title, global_step=self.global_step)
                        plt.close(f)

                if 'alpha_beta' in plot_type:
                    if self.model.pooler in ['baypool']:
                        title = f'alpha_beta'
                        if istest:
                            title += '_test'
                        f, axes = plt.subplots(2, 1, figsize=(12, 6)) 
                        alpha = self.model.pool.alpa_tilde.detach().cpu().numpy().mean(axis=(0,1))
                        beta = self.model.pool.beta_tilde.detach().cpu().numpy().mean(axis=(0,1))
                        axes[0].bar(np.arange(len(alpha)), alpha, color='tab:blue', alpha=0.5)
                        axes[0].set_title('Alpha')              
                        axes[1].bar(np.arange(len(beta)), beta, color='tab:red', alpha=0.5)
                        axes[1].set_title('Beta')
                        plt.tight_layout()
                        self.logger.log_figure(f, title, global_step=self.global_step)
                        plt.close(f)
      

    def __init__(self,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class: Optional[Type] = None,
                 scheduler_kwargs: Optional[Mapping] = None,
                 log_lr: bool = True,
                 log_grad_norm: bool = False,
                 ):
        super().__init__()
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs or dict()
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or dict()
        self.log_lr = log_lr
        self.log_grad_norm = log_grad_norm

        
    def configure_optimizers(self):
        """
        🛠️ Configure optimizer and scheduler
        """
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg['optimizer'] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop('monitor', None)
            scheduler = self.scheduler_class(optimizer,
                                             **self.scheduler_kwargs)
            cfg['lr_scheduler'] = scheduler
            if metric is not None:
                cfg['monitor'] = metric
        return cfg


    def on_before_optimizer_step(self, optimizer):
        """
        📏 Log gradients norm
        """
        if self.log_grad_norm:
            self.log_dict(grad_norm(self, norm_type=2))


    def on_train_epoch_start(self) -> None:
        """
        ⌚ Log learning rate at the start of each epoch
        """
        if self.log_lr:
            optimizers = self.optimizers()
            if isinstance(optimizers, list):
                for i, optimizer in enumerate(optimizers):
                    lr = optimizer.optimizer.param_groups[0]['lr']
                    self.log(f'lr_{i}', lr, on_step=False, on_epoch=True,
                             logger=True, prog_bar=False, batch_size=1)
            else:
                lr = optimizers.optimizer.param_groups[0]['lr']
                self.log(f'lr', lr, on_step=False, on_epoch=True,
                         logger=True, prog_bar=False, batch_size=1)