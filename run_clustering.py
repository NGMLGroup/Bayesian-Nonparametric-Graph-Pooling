import torch
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import DictConfig, OmegaConf

# Local imports
from source.data import PyGSPDataset
from source.pl_modules import ClusterModule
from source.models import ClusterModel
from source.utils import (register_resolvers, 
                          reduce_precision, 
                          find_devices,
                          NormalizeAdjSparse_with_ea,
                          SortNodes,
                          CoefficientScheduler,
                          CustomTensorBoardLogger)

register_resolvers()
reduce_precision()


@hydra.main(version_base=None, config_path="config", config_name="run_clustering")
def run(cfg : DictConfig) -> float:

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ### 📊 Load data
    trans = Compose([SortNodes(), NormalizeAdjSparse_with_ea(delta=0.85)]) if cfg.pooler.name == 'bnpool' else SortNodes()
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
                                     kwargs=cfg.dataset.params, force_reload=cfg.dataset.hparams.reload,
                                     pre_transform=trans)
        num_classes = cfg.architecture.hparams.pool_ratio 
    else:
        raise ValueError(f"Dataset {cfg.dataset.family} not recognized")
    
    num_clusters = cfg.pooler.hparams.n_clusters if cfg.pooler.name == 'bnpool' else num_classes
      
    data_loader = DataLoader(torch_dataset, batch_size=cfg.batch_size, shuffle=False)


    ### 🧠 Load the model
    torch_model = ClusterModel(
        in_channels=torch_dataset.num_features,                         # Size of node features
        num_layers_pre=cfg.architecture.hparams.num_layers_pre,         # Number of GIN layers before pooling
        hidden_channels=cfg.architecture.hparams.hidden_channels,       # Dimensionality of node embeddings
        activation=cfg.architecture.hparams.activation,                 # Activation of the MLP in GIN 
        use_cache=cfg.architecture.hparams.use_cache,                   # Cache computation of dense adjacency
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
        log_grad_norm=cfg.log_grad_norm,
        plot_dict=dict(cfg.plot_preds_at_epoch))


    ### 🪵 Logger
    if cfg.get('logger').get('backend') is None:
        logger = None
    elif cfg.logger.backend == 'tensorboard':
        logger = CustomTensorBoardLogger(save_dir=cfg.logger.logdir, name=None, version='')
        logger.cfg = cfg
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
    trainer = pl.Trainer(
        logger=logger,
        callbacks=cb, 
        devices=find_devices(1),
        max_epochs=cfg.epochs, 
        gradient_clip_val=cfg.clip_val,
        accelerator='gpu',
        )
    trainer.fit(lightning_model, data_loader)

    if cfg.callbacks.checkpoints:
        trainer.test(lightning_model, data_loader, ckpt_path='best')
    else:
        trainer.test(lightning_model, data_loader)

    logger.finalize('success')

    return trainer.callback_metrics["test_loss"].item()

if __name__ == "__main__":
    run()