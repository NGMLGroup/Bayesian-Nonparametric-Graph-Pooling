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
                 ):
        super().__init__(optim_class, optim_kwargs, scheduler_class, scheduler_kwargs,
                         log_lr, log_grad_norm)

        self.model = model 
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
        s, aux_loss, loss_d = self.model(data)

        return s, aux_loss, loss_d


    def training_step(self, batch, batch_idx):
        """
        🐾
        """
        y = batch.y
        s, aux_loss, loss_d = self.forward(batch)
        self.log('train_loss', aux_loss, on_step=False, on_epoch=True, 
                 prog_bar=False, batch_size=batch.batch_size)
        if loss_d is not None and self.model.pooler == 'bnpool':
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

        return {'loss':aux_loss}
    

    def test_step(self, batch, batch_idx):
        """
        🧪
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        y = batch.y
        s, aux_loss, _ = self.forward(batch)

        # Log metrics
        self.log('test_loss', aux_loss, batch_size=batch.batch_size)
        self.test_metrics.update(s[0].argmax(axis=-1).detach(), y.int())
        self.test_fuzzy_metrics.update(s[0], y.long())
        self.log('test_NMI', self.test_metrics['NMI'], batch_size=batch.batch_size)
        self.log('test_FuzzyClusterCosine', self.test_fuzzy_metrics['FuzzyClusterCosine'], batch_size=batch.batch_size)
        clf = LogisticRegression(random_state=0).fit(s[0].detach().cpu().numpy(), y.cpu().numpy())
        y_pred = clf.predict(s[0].detach().cpu().numpy())
        clf_acc = accuracy_score(y.cpu().numpy(), y_pred)
        self.log('CLF_acc', clf_acc, batch_size=batch.batch_size)
