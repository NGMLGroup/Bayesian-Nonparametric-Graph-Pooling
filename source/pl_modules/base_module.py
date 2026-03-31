from typing import Optional, Mapping, Type
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm

class BaseModule(pl.LightningModule):
    """
    🧱 Base Lightning Module class
    """

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
