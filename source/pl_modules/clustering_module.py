from typing import Optional, Mapping, Type
import torch
import torchmetrics
from torchmetrics.clustering import NormalizedMutualInfoScore

# Local imports
from .base_module import BaseModule
from source.utils.metrics import FuzzyClusterCosine


class ClusterModule(BaseModule):
    """
    Lightning module to perform clustering with graph pooling 🎱
    """
    def __init__(self,
                 model: Optional[torch.nn.Module] = None,
                 num_classes: int = None,
                 num_clusters: int = None,
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
        self.train_metrics = torchmetrics.MetricCollection({
            'NMI': NormalizedMutualInfoScore(),
        })
        self.train_fuzzy_metrics = torchmetrics.MetricCollection({
            'FuzzyClusterCosine': FuzzyClusterCosine(num_classes=num_classes, num_clusters=num_clusters)
        })
        self.test_metrics = torchmetrics.MetricCollection({
            'NMI': NormalizedMutualInfoScore(),
        })
        self.test_fuzzy_metrics = torchmetrics.MetricCollection({
            'FuzzyClusterCosine': FuzzyClusterCosine(num_classes=num_classes, num_clusters=num_clusters)
        })


    def forward(self, data):
        """
        ⏩ 
        """
        x_pool, adj_pool, batch_pool, s, aux_loss, loss_d = self.model(data)

        return x_pool, adj_pool, batch_pool, s, aux_loss, loss_d


    def training_step(self, batch, batch_idx):
        """
        🐾
        """
        y = batch.y
        _, adj_pool, batch_pool, s, aux_loss, loss_d = self.forward(batch)
        self.log('train_loss', aux_loss, on_step=False, on_epoch=True, 
                 prog_bar=False, batch_size=batch.batch_size)
        if loss_d is not None:
            self.log('train_quality', loss_d['quality'], on_step=False, on_epoch=True, 
                    prog_bar=True, batch_size=batch.batch_size)
            self.log('train_kl', loss_d['kl'], on_step=False, on_epoch=True, 
                    prog_bar=True, batch_size=batch.batch_size)
            self.log('K_prior', loss_d['K_prior'], on_step=False, on_epoch=True,
                     prog_bar=False, batch_size=batch.batch_size)
        self.train_metrics.update(s[0].argmax(axis=-1).detach(), y.int())
        self.train_fuzzy_metrics.update(s[0], y.long())
        self.log('NMI', self.train_metrics['NMI'], on_step=False, on_epoch=True, 
                 prog_bar=True, batch_size=batch.batch_size)
        self.log('FuzzyClusterCosine', self.train_fuzzy_metrics['FuzzyClusterCosine'], on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=batch.batch_size)

        # Log images and artifacts
        if 'train' in self.plot_preds_at_epoch['set']:
            if self.logger is not None:
                if self.logger.cfg.logger.backend in ['tensorboard']:
                    self.maybe_log_stuff(
                        batch=batch, 
                        batch_idx=batch_idx, 
                        pooled_adj=adj_pool, 
                        pooled_batch=batch_pool, 
                        s=s, 
                        plot_type=self.plot_preds_at_epoch['types'], 
                        istest=False)

        return {'loss':aux_loss}
    

    def test_step(self, batch, batch_idx):
        """
        🧪
        """
        y = batch.y
        _, adj_pool, batch_pool, s, aux_loss, _ = self.forward(batch)

        # Log metrics
        self.log('test_loss', aux_loss, batch_size=batch.batch_size)
        self.test_metrics.update(s[0].argmax(axis=-1).detach(), y.int())
        self.test_fuzzy_metrics.update(s[0], y.long())
        self.log('test_NMI', self.test_metrics['NMI'], batch_size=batch.batch_size)
        self.log('test_FuzzyClusterCosine', self.test_fuzzy_metrics['FuzzyClusterCosine'], batch_size=batch.batch_size)
        
        # Fit a logistic regression classifier on s to predict y
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        clf = LogisticRegression(random_state=0).fit(s[0].detach().cpu().numpy(), y.cpu().numpy())
        y_pred = clf.predict(s[0].detach().cpu().numpy())
        acc = accuracy_score(y.cpu().numpy(), y_pred)
        self.log('CLF_acc', acc)

        # Log images and artifacts
        if 'test' in self.plot_preds_at_epoch['set']:
            if self.logger is not None:
                if self.logger.cfg.logger.backend in ['tensorboard']:
                    self.maybe_log_stuff(
                        batch=batch, 
                        batch_idx=batch_idx, 
                        pooled_adj=adj_pool, 
                        pooled_batch=batch_pool, 
                        s=s, 
                        plot_type=self.plot_preds_at_epoch['types'], 
                        istest=True)