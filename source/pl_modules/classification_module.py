from typing import Optional, Mapping, Type
import torch
import torchmetrics
from torchmetrics.classification import Accuracy, MulticlassF1Score, MulticlassAUROC
from source.utils import NonEmptyClusterTracker

from .base_module import BaseModule


class ClassificationModule(BaseModule):
    def __init__(self,
                 model: Optional[torch.nn.Module] = None,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class: Optional[Type] = None,
                 scheduler_kwargs: Optional[Mapping] = None,
                 log_lr: bool = True,
                 log_grad_norm: bool = False,
                 plot_dict: Optional[Mapping] = None
                 ):
        super().__init__(optim_class, optim_kwargs, scheduler_class, scheduler_kwargs,
                         log_lr, log_grad_norm)

        self.model = model 
        self.plot_preds_at_epoch = plot_dict
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_metrics = torchmetrics.MetricCollection({
            'train_acc': Accuracy(task='multiclass', num_classes=model.num_classes),
            'train_f1': MulticlassF1Score(num_classes=model.num_classes, average='macro'),
            'train_auroc': MulticlassAUROC(num_classes=model.num_classes),
            })
        self.val_metrics = torchmetrics.MetricCollection({
            'val_acc': Accuracy(task='multiclass', num_classes=model.num_classes),
            'val_f1': MulticlassF1Score(num_classes=model.num_classes, average='macro'),
            'val_auroc': MulticlassAUROC(num_classes=model.num_classes),
            })
        self.test_metrics = torchmetrics.MetricCollection({
            'test_acc': Accuracy(task='multiclass', num_classes=model.num_classes),
            'test_f1': MulticlassF1Score(num_classes=model.num_classes, average='macro'),
            'test_auroc': MulticlassAUROC(num_classes=model.num_classes),
            })
        
        if self.model.pooler == 'bnpool': # Tracks the number of non-empty clusters
            self.non_empty_tracker_tr = NonEmptyClusterTracker() 
            self.non_empty_tracker_val = NonEmptyClusterTracker() 
            self.non_empty_tracker_test = NonEmptyClusterTracker() 
        

    def maybe_log_histogram(self, counts, istest=False, n_bins=50):
        """
        Log histogram of the number of non-empty clusters.
        This function is here instead of in maybe_log_stuff from the base class
        because is called at epoch end
        """

        if self.plot_preds_at_epoch is not None:
            every = self.plot_preds_at_epoch.get('every', 1)  

            if self.current_epoch%every==0 or istest:

                import matplotlib.pyplot as plt
                import numpy as np
                f, ax = plt.subplots(figsize=(10, 4))
                ax.hist(counts.cpu().numpy(), 
                        bins=n_bins,
                        range=(0, n_bins),
                        edgecolor='black',
                        linewidth=1.2,
                        color='cornflowerblue',
                        alpha=0.7,
                        rwidth=0.85)

                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlabel('Number of Non-Empty Clusters')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Non-Empty Clusters')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.2)
                ax.spines['bottom'].set_linewidth(1.2)

                ax.set_xticks(np.arange(n_bins) + 0.5)
                ax.set_xticklabels([str(i) if i % 5 == 0 else '' for i in np.arange(n_bins)])

                plt.tight_layout()
                self.logger.log_figure(f, 'nonempty_clust', global_step=self.global_step)
                plt.close(f)


    def forward(self, data):
        """
        ⏩ 
        """
        logits, pooled_adj, edge_weight, pooled_batch, s, aux_loss, loss_d, nonempty_clust = self.model(data)

        return logits, pooled_adj, pooled_batch, s, aux_loss, loss_d, nonempty_clust
    

    def training_step(self, batch, batch_idx):
        """
        🐾
        """
        logits, pooled_adj, pooled_batch, s, aux_loss, loss_d, nonempty_clust = self.forward(batch)
        clf_loss = self.loss(logits, batch.y)
        loss = clf_loss + aux_loss

        # Log losses and metrics
        self.train_metrics.update(logits, batch.y)
        self.log('train_clf_loss', clf_loss, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_aux_loss', aux_loss, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=False)
        if self.model.pooler == 'bnpool':
            self.log('eta', self.model.pool.eta, batch_size=batch.y.size(0),
                    on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_quality', loss_d['quality'], on_step=False, on_epoch=True, 
                    prog_bar=True, batch_size=batch.batch_size)
            self.log('train_kl', loss_d['kl'], on_step=False, on_epoch=True, 
                    prog_bar=True, batch_size=batch.batch_size)
            self.log('K_prior', loss_d['K_prior'], on_step=False, on_epoch=True,
                     prog_bar=True, batch_size=batch.batch_size)
            self.non_empty_tracker_tr.update(nonempty_clust.sum(axis=-1))
        
        # Log images and artifacts
        if 'train' in self.plot_preds_at_epoch['set']:
            if self.logger is not None:
                if self.logger.cfg.logger.backend in ['tensorboard']:
                    self.maybe_log_stuff(batch=batch, batch_idx=batch_idx, 
                                         pooled_adj=pooled_adj, pooled_batch=pooled_batch, 
                                         s=s, plot_type=self.plot_preds_at_epoch['types'], 
                                         istest=False)

        return {'loss':loss}
    

    def on_train_epoch_end(self):
        """
        🏁
        """
        train_ = self.train_metrics.compute()
        self.log('train_acc', train_['train_acc'])
        self.log('train_f1', train_['train_f1'])
        self.log('train_auroc', train_['train_auroc'])
        self.train_metrics.reset()

        if self.model.pooler == 'bnpool':
            counts = self.non_empty_tracker_tr.compute()
            self.non_empty_tracker_tr.reset()

            if 'train' in self.plot_preds_at_epoch['set']:
                if self.logger is not None:
                    if self.logger.cfg.logger.backend in ['tensorboard']:
                        if 'nonempty_clust' in self.plot_preds_at_epoch['types']:
                            self.maybe_log_histogram(counts, istest=False)


    def validation_step(self, batch, batch_idx):
        """
        🐾
        """
        logits, pooled_adj, pooled_batch, s, aux_loss, loss_d, nonempty_clust = self.forward(batch)
        clf_loss = self.loss(logits, batch.y)
        loss = clf_loss + aux_loss

        # Log losses and metrics
        self.val_metrics.update(logits, batch.y)
        self.log('val_clf_loss', clf_loss, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_aux_loss', aux_loss, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, batch_size=batch.y.size(0),
                 on_step=False, on_epoch=True, prog_bar=False)
        if self.model.pooler == 'bnpool':
            self.log('val_quality', loss_d['quality'], on_step=False, on_epoch=True, 
                    prog_bar=True, batch_size=batch.batch_size)
            self.log('val_kl', loss_d['kl'], on_step=False, on_epoch=True,
                    prog_bar=True, batch_size=batch.batch_size)
            self.log('K_prior', loss_d['K_prior'], on_step=False, on_epoch=True,
                     prog_bar=True, batch_size=batch.batch_size)
            self.non_empty_tracker_val.update(nonempty_clust.sum(axis=-1))

        # Log images and artifacts
        if 'val' in self.plot_preds_at_epoch['set']:
            if self.logger is not None:
                if self.logger.cfg.logger.backend in ['tensorboard']:
                    self.maybe_log_stuff(batch=batch, batch_idx=batch_idx, 
                                         pooled_adj=pooled_adj, pooled_batch=pooled_batch, 
                                         s=s, plot_type=self.plot_preds_at_epoch['types'], 
                                         istest=False)

        return {'val_loss':loss}
    

    def on_validation_epoch_end(self):
        """
        🏁
        """
        val_ = self.val_metrics.compute()
        self.log('val_acc', val_['val_acc'])
        self.log('val_f1', val_['val_f1'])
        self.log('val_auroc', val_['val_auroc'])
        self.val_metrics.reset()

        if self.model.pooler == 'bnpool':
            counts = self.non_empty_tracker_val.compute()
            self.non_empty_tracker_val.reset()
            if 'val' in self.plot_preds_at_epoch['set']:
                if self.logger is not None:
                    if self.logger.cfg.logger.backend in ['tensorboard']:
                        if 'nonempty_clust' in self.plot_preds_at_epoch['types']:
                            self.maybe_log_histogram(counts, istest=False)
        

    def test_step(self, batch, batch_idx):
        """
        🧪
        """
        logits, pooled_adj, pooled_batch, s, aux_loss, _, nonempty_clust = self.forward(batch)
        clf_loss = self.loss(logits, batch.y)
        loss = clf_loss + aux_loss

        # Log losses and metrics
        self.test_metrics.update(logits, batch.y)
        self.log('test_clf_loss', clf_loss, batch_size=batch.y.size(0))
        self.log('test_aux_loss', aux_loss, batch_size=batch.y.size(0))
        self.log('test_loss', loss, batch_size=batch.y.size(0))
        if self.model.pooler == 'bnpool':
            self.non_empty_tracker_test.update(nonempty_clust.sum(axis=-1))
        
        # Log images and artifacts
        if 'test' in self.plot_preds_at_epoch['set']:
            if self.logger is not None:
                if self.logger.cfg.logger.backend in ['tensorboard']:
                    self.maybe_log_stuff(batch=batch, batch_idx=batch_idx, 
                                         pooled_adj=pooled_adj, pooled_batch=pooled_batch, 
                                         s=s, plot_type=self.plot_preds_at_epoch['types'], 
                                         istest=True)

        return {'test_loss':loss}
    

    def on_test_epoch_end(self):
        """
        🏁
        """
        test_ = self.test_metrics.compute()
        self.log('test_acc', test_['test_acc'])
        self.log('test_f1', test_['test_f1'])
        self.log('test_auroc', test_['test_auroc'])

        if self.model.pooler == 'bnpool':
            counts = self.non_empty_tracker_test.compute()
            self.non_empty_tracker_test.reset()
            if 'test' in self.plot_preds_at_epoch['set']:
                if self.logger is not None:
                    if self.logger.cfg.logger.backend in ['tensorboard']:
                        if 'nonempty_clust' in self.plot_preds_at_epoch['types']:
                            self.maybe_log_histogram(counts, istest=True)