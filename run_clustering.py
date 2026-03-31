import torch
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from tgp.poolers import pooler_map

# Local imports
from source.data import PyGSPDataset
from source.pl_modules import ClusterModule
from source.models import ClusterModel
from source.utils import (register_resolvers, 
                          reduce_precision, 
                          find_devices,
                          SortNodes,
                          CoefficientScheduler)

register_resolvers()
reduce_precision()


@hydra.main(version_base=None, config_path="config", config_name="run_clustering")
def run(cfg : DictConfig) -> float:

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ### 📊 Load data
    pooler_transform = pooler_map[cfg.pooler.name].data_transforms()
    trans = Compose([SortNodes(), pooler_transform]) if pooler_transform is not None else SortNodes()
    if cfg.dataset.family=='Planetoid':
        torch_dataset = Planetoid(root='data/', name=cfg.dataset.name,
                                  split=cfg.dataset.hparams.split,
                                  pre_transform=trans, force_reload=True)
        num_classes = torch_dataset.num_classes
    elif cfg.dataset.family=='CitationFull':
        torch_dataset = CitationFull(root='data/', name=cfg.dataset.name,
                                     pre_transform=trans, force_reload=True)
        num_classes = torch_dataset.num_classes
    elif cfg.dataset.family=='PyGSPDataset':
        torch_dataset = PyGSPDataset(root='data/PyGSP', name=cfg.dataset.name, 
                                     kwargs=cfg.dataset.params, force_reload=True,
                                     pre_transform=trans)
        num_classes = cfg.architecture.hparams.pool_ratio 
    else:
        raise ValueError(f"Dataset {cfg.dataset.family} not recognized")
    
    pooler_hparams = {} if cfg.pooler.hparams is None else dict(cfg.pooler.hparams)
    num_clusters = pooler_hparams.get('k', num_classes)
      
    data_loader = DataLoader(torch_dataset, batch_size=cfg.batch_size, shuffle=False)


    ### 🧠 Load the model
    torch_model = ClusterModel(
        in_channels=torch_dataset.num_features,                         # Size of node features
        num_layers_pre=cfg.architecture.hparams.num_layers_pre,         # Number of GIN layers before pooling
        hidden_channels=cfg.architecture.hparams.hidden_channels,       # Dimensionality of node embeddings
        activation=cfg.architecture.hparams.activation,                 # Activation of the MLP in GIN 
        pooler=cfg.pooler.name,                                         # Pooling method
        pool_kwargs=cfg.pooler.hparams,                                 # Pooling method kwargs
        pooled_nodes=num_classes,                                       # Number of nodes after pooling
        )


    ### 📈 Optimizer scheduler
    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None


    ### ⚡ Lightning module
    lightning_model = ClusterModule(
        model=torch_model,
        num_classes=num_classes,
        num_clusters=num_clusters,
        optim_class=getattr(torch.optim, 'Adam'),
        optim_kwargs=dict(cfg.optimizer.hparams),
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        log_lr=cfg.log_lr,
        log_grad_norm=cfg.log_grad_norm)


    ### 🪵 Logger
    if cfg.get('logger').get('backend') is None:
        logger = None
    elif cfg.logger.backend == 'tensorboard':
        logger = TensorBoardLogger(save_dir=cfg.logger.logdir, name=None, version='')
    else:
        raise NotImplementedError("Logger backend not supported.")
    

    ### 📞 Callbacks
    cb = []
    if cfg.callbacks.early_stop:
        early_stop_callback = EarlyStopping(
            monitor=cfg.callbacks.monitor,
            patience=cfg.callbacks.patience,
            mode=cfg.callbacks.mode
        )
        cb.append(early_stop_callback)
    
    if cfg.callbacks.checkpoints:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor=cfg.callbacks.monitor,
            mode=cfg.callbacks.mode,
            dirpath=cfg.logger.logdir+"/checkpoints/", 
            filename=cfg.architecture.name + "_" + cfg.pooler.name + "___{epoch:03d}-{NMI:e}",
        )
        cb.append(checkpoint_callback)

    if cfg.callbacks.params_scheduling.activate:
        parameter_scheduler = CoefficientScheduler(
            epochs=cfg.callbacks.params_scheduling.epochs,
            first_eta=cfg.callbacks.params_scheduling.first_eta,
            last_eta=cfg.callbacks.params_scheduling.last_eta,
            mode=cfg.callbacks.params_scheduling.mode)
        cb.append(parameter_scheduler)


    ### 🚀 Trainer
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = find_devices(1) if accelerator == 'gpu' else 1
    trainer = pl.Trainer(
        logger=logger,
        callbacks=cb, 
        devices=devices,
        max_epochs=cfg.epochs, 
        gradient_clip_val=cfg.clip_val,
        accelerator=accelerator,
        )
    trainer.fit(lightning_model, data_loader)

    if cfg.callbacks.checkpoints:
        trainer.test(lightning_model, data_loader, ckpt_path='best')
    else:
        trainer.test(lightning_model, data_loader)

    if logger is not None:
        logger.finalize('success')

    return trainer.callback_metrics["test_loss"].item()

if __name__ == "__main__":
    run()
