from typing import Mapping, Optional, Type

import torch
import torchmetrics
from torchmetrics.classification import (
    Accuracy,
    AveragePrecision,
    MulticlassAUROC,
    MulticlassF1Score,
)

from source.utils import ClusterOccupancyTracker, NonEmptyClusterTracker

from .base_module import BaseModule


class BaseClassificationModule(BaseModule):
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        optim_class: Optional[Type] = None,
        optim_kwargs: Optional[Mapping] = None,
        scheduler_class: Optional[Type] = None,
        scheduler_kwargs: Optional[Mapping] = None,
        log_lr: bool = True,
        log_grad_norm: bool = False,
    ):
        super().__init__(
            optim_class,
            optim_kwargs,
            scheduler_class,
            scheduler_kwargs,
            log_lr,
            log_grad_norm,
        )

        self.model = model
        self.init_metrics()
        self.track_cluster_occupancy = self.model.pooler == 'bnpool' and self.model.pool.batched

        if self.track_cluster_occupancy:
            n_clusters = self.model.pool.k
            self.non_empty_tracker_tr = NonEmptyClusterTracker(n_clusters)
            self.non_empty_tracker_val = NonEmptyClusterTracker(n_clusters)
            self.non_empty_tracker_test = NonEmptyClusterTracker(n_clusters)
            self.clust_occup_tracker_tr = ClusterOccupancyTracker(n_clusters)
            self.clust_occup_tracker_val = ClusterOccupancyTracker(n_clusters)
            self.clust_occup_tracker_test = ClusterOccupancyTracker(n_clusters)

    def init_metrics(self):
        raise NotImplementedError

    def compute_clf_loss(self, logits, labels):
        return self.loss(logits, labels)

    def update_metrics(self, stage, logits, labels):
        metrics = getattr(self, f'{stage}_metrics', None)
        if metrics is not None:
            metrics.update(logits, labels)

    def log_metrics_epoch(self, stage):
        metrics = getattr(self, f'{stage}_metrics', None)
        if metrics is None:
            return

        values = metrics.compute()
        for name, value in values.items():
            self.log(name, value)
        metrics.reset()

    def _log_cluster_metrics(self, stage, counts, occupancies):
        if counts.numel() == 0:
            return

        self.log(
            f'{stage}_nonempty_clust_mean',
            counts.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f'{stage}_nonempty_clust_std',
            counts.std(unbiased=False),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f'{stage}_clust_occupancy',
            (occupancies > (1.0 / self.model.pool.k)).sum().float(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def _track_clusters(self, stage, clust_occup):
        stage_key = 'tr' if stage == 'train' else stage
        getattr(self, f'non_empty_tracker_{stage_key}').update(clust_occup)
        getattr(self, f'clust_occup_tracker_{stage_key}').update(clust_occup)

    def _flush_cluster_metrics(self, stage):
        stage_key = 'tr' if stage == 'train' else stage
        counts = getattr(self, f'non_empty_tracker_{stage_key}').compute()
        occupancies = getattr(self, f'clust_occup_tracker_{stage_key}').compute()
        getattr(self, f'non_empty_tracker_{stage_key}').reset()
        getattr(self, f'clust_occup_tracker_{stage_key}').reset()
        self._log_cluster_metrics(stage, counts, occupancies)

    def forward(self, data):
        logits, aux_loss, loss_d, clust_occup = self.model(data)
        return logits, aux_loss, loss_d, clust_occup

    def _shared_step(self, batch, batch_idx, stage):
        logits, aux_loss, loss_d, clust_occup = self.forward(batch)
        clf_loss = self.compute_clf_loss(logits, batch.y)
        loss = clf_loss + aux_loss

        self.log(
            f'{stage}_clf_loss',
            clf_loss,
            batch_size=batch.y.size(0),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f'{stage}_aux_loss',
            aux_loss,
            batch_size=batch.y.size(0),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f'{stage}_loss',
            loss,
            batch_size=batch.y.size(0),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.update_metrics(stage, logits, batch.y)

        if self.model.pooler == 'bnpool':
            if stage == 'train':
                self.log(
                    'eta',
                    self.model.pool.eta,
                    batch_size=batch.y.size(0),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
            if loss_d is not None and stage in ['train', 'val']:
                self.log(
                    f'{stage}_quality',
                    loss_d['quality'],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch.batch_size,
                )
                self.log(
                    f'{stage}_kl',
                    loss_d['kl'],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch.batch_size,
                )
                self.log(
                    'K_prior',
                    loss_d['K_prior'],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch.batch_size,
                )
            if self.track_cluster_occupancy and clust_occup is not None:
                self._track_clusters(stage, clust_occup)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, 'train')
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, 'val')
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, 'test')
        return {'test_loss': loss}

    def on_train_epoch_end(self):
        self.log_metrics_epoch('train')
        if self.track_cluster_occupancy:
            self._flush_cluster_metrics('train')

    def on_validation_epoch_end(self):
        self.log_metrics_epoch('val')
        if self.track_cluster_occupancy:
            self._flush_cluster_metrics('val')

    def on_test_epoch_end(self):
        self.log_metrics_epoch('test')
        if self.track_cluster_occupancy:
            self._flush_cluster_metrics('test')


class ClassificationModule(BaseClassificationModule):
    def init_metrics(self):
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_metrics = torchmetrics.MetricCollection({
            'train_acc': Accuracy(task='multiclass', num_classes=self.model.num_classes),
            'train_f1': MulticlassF1Score(num_classes=self.model.num_classes, average='macro'),
            'train_auroc': MulticlassAUROC(num_classes=self.model.num_classes),
        })
        self.val_metrics = torchmetrics.MetricCollection({
            'val_acc': Accuracy(task='multiclass', num_classes=self.model.num_classes),
            'val_f1': MulticlassF1Score(num_classes=self.model.num_classes, average='macro'),
            'val_auroc': MulticlassAUROC(num_classes=self.model.num_classes),
        })
        self.test_metrics = torchmetrics.MetricCollection({
            'test_acc': Accuracy(task='multiclass', num_classes=self.model.num_classes),
            'test_f1': MulticlassF1Score(num_classes=self.model.num_classes, average='macro'),
            'test_auroc': MulticlassAUROC(num_classes=self.model.num_classes),
        })


class MultiClassificationModule(BaseClassificationModule):
    def init_metrics(self):
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.train_metrics = torchmetrics.MetricCollection({
            'train_ap': AveragePrecision(
                task='multilabel',
                num_labels=self.model.num_classes,
                average='macro',
            ),
        })
        self.val_metrics = torchmetrics.MetricCollection({
            'val_ap': AveragePrecision(
                task='multilabel',
                num_labels=self.model.num_classes,
                average='macro',
            ),
        })
        self.test_metrics = torchmetrics.MetricCollection({
            'test_ap': AveragePrecision(
                task='multilabel',
                num_labels=self.model.num_classes,
                average='macro',
            ),
        })

    def _filter_valid_rows(self, logits, labels):
        labels = labels.float()
        if labels.dim() == 1:
            valid_rows = ~torch.isnan(labels)
        else:
            valid_rows = ~torch.isnan(labels).any(dim=-1)
        return logits[valid_rows], labels[valid_rows]

    def compute_clf_loss(self, logits, labels):
        logits, labels = self._filter_valid_rows(logits, labels)
        if logits.numel() == 0:
            return logits.sum() * 0
        return self.loss(logits, labels)

    def update_metrics(self, stage, logits, labels):
        logits, labels = self._filter_valid_rows(logits, labels)
        if logits.numel() == 0:
            return
        super().update_metrics(stage, logits, labels.long())


class RegressionModule(BaseClassificationModule):
    def init_metrics(self):
        self.loss = torch.nn.MSELoss()
        self.train_metrics = torchmetrics.MetricCollection({
            'train_mae': torchmetrics.MeanAbsoluteError(),
        })
        self.val_metrics = torchmetrics.MetricCollection({
            'val_mae': torchmetrics.MeanAbsoluteError(),
        })
        self.test_metrics = torchmetrics.MetricCollection({
            'test_mae': torchmetrics.MeanAbsoluteError(),
        })

    def compute_clf_loss(self, logits, labels):
        return self.loss(logits, labels.float())

    def update_metrics(self, stage, logits, labels):
        super().update_metrics(stage, logits, labels.float())
