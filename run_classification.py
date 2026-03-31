import torch
import torch_geometric
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from tgp.poolers import pooler_map

# Local imports
from source.data import (
    BenchHardDataModule, 
    MultipartiteDataModule,
    TUDataModule, 
    OGBDataModule,
    LRGBDataModule,
)
from source.pl_modules import ClassificationModule, MultiClassificationModule, RegressionModule
from source.models import ClassificationModel, MolhivModel
from source.utils import (register_resolvers, 
                          reduce_precision, 
                          find_devices,
                          CoefficientScheduler)

register_resolvers()
reduce_precision()


ATOM_BOND_ENCODER_DATASETS = {'ogbg-molhiv', 'peptides-struct'}


@hydra.main(version_base=None, config_path="config", config_name="run_classification")
def run(cfg : DictConfig) -> float:

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ### 🌱 Seed everything
    if 'seed' in cfg.dataset.hparams:
        print(f"Setting seed to {cfg.dataset.hparams.seed}")
        torch_geometric.seed.seed_everything(cfg.dataset.hparams.seed)

    ### 📊 Load data
    pre_transform = pooler_map[cfg.pooler.name].data_transforms()
    if cfg.dataset.get('family') is not None:
        if cfg.dataset.family in ['TUDataset']:
            data_module = TUDataModule(cfg.dataset.name, cfg.dataset.hparams, pre_transform=pre_transform)
        elif cfg.dataset.family in ['OGBDataset']:
            data_module = OGBDataModule(cfg.dataset.name, cfg.dataset.hparams, pre_transform=pre_transform)
        elif cfg.dataset.family in ['LRGBDataset']:
            data_module = LRGBDataModule(cfg.dataset.name, cfg.dataset.hparams, pre_transform=pre_transform)
        else:
            raise ValueError(f"Dataset family {cfg.dataset.family} not recognized")
    else:
        if cfg.dataset.name in ['BenchHard']:
            data_module = BenchHardDataModule(cfg.dataset.hparams, pre_transform=pre_transform)
        elif cfg.dataset.name in ['Multipartite']:
            data_module = MultipartiteDataModule(cfg.dataset.hparams, pre_transform=pre_transform)
        else:
            raise ValueError(f"Dataset {cfg.dataset.name} not recognized")
        
    ### 🧠 Load the model
    model_class = MolhivModel if cfg.dataset.name in ATOM_BOND_ENCODER_DATASETS else ClassificationModel
    torch_model = model_class(
        in_channels=data_module.num_features,
        out_channels=data_module.num_classes,
        edge_channels=data_module.num_edge_features,
        num_layers_pre=cfg.architecture.hparams.num_layers_pre,
        num_layers_post=cfg.architecture.hparams.num_layers_post,
        hidden_channels=cfg.architecture.hparams.hidden_channels,
        activation=cfg.architecture.hparams.activation,
        dropout=cfg.architecture.hparams.dropout,
        pooler=cfg.pooler.name,
        pool_kwargs=cfg.pooler.hparams,
        pooled_nodes=int(data_module.avg_nodes * cfg.architecture.hparams.pool_ratio),
        use_gine=bool(data_module.num_edge_features),
    )

    ### 📈 Optimizer scheduler
    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    ### ⚡ Lightning module
    if cfg.dataset.name in ['peptides-func']:
        module_class = MultiClassificationModule
    elif cfg.dataset.name in ['peptides-struct']:
        module_class = RegressionModule
    else:
        module_class = ClassificationModule

    lightning_model = module_class(
        model=torch_model,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
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
            filename=cfg.architecture.name + "_" + cfg.pooler.name + "___{epoch:03d}-{cfg.callbacks.monitor:e}",
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
    devices = find_devices(2) if accelerator == 'gpu' else 1
    trainer = pl.Trainer(
        logger=logger,
        callbacks=cb, 
        devices=devices,
        max_epochs=cfg.epochs, 
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        gradient_clip_val=cfg.clip_val,
        accelerator=accelerator,
        )
    
    trainer.fit(lightning_model, data_module.train_dataloader(), data_module.val_dataloader())
    val_loss = trainer.callback_metrics[cfg.callbacks.monitor].item()

    if cfg.callbacks.checkpoints:
        trainer.test(lightning_model, data_module.test_dataloader(), ckpt_path='best')
    else:
        trainer.test(lightning_model, data_module.test_dataloader())
        
    if logger is not None:
        logger.finalize('success')

    return val_loss

if __name__ == "__main__":
    run()
