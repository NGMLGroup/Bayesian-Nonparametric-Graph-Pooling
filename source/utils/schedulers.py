import torch
import pytorch_lightning as pl


class LinearScheduler:
    def __init__(self, first_val=1, last_val=0, epochs=100):
        self.first_val = first_val
        self.last_val = last_val
        self.epochs = epochs

    def __call__(self, epoch):
        epoch = min(self.epochs, epoch)
        return self.first_val + (self.last_val - self.first_val) * epoch / self.epochs


class CosineScheduler:
    def __init__(self, first_val=1, last_val=0, epochs=100):
        self.first_val = first_val
        self.last_val = last_val
        self.epochs = epochs

    def __call__(self, epoch):
        epoch = torch.tensor(min(self.epochs, epoch))
        return self.first_val + 0.5 * (self.last_val - self.first_val) * (1 - torch.cos(epoch * 3.1415 / self.epochs))


class CoefficientScheduler(pl.Callback):
    def __init__(self, epochs, first_eta=1, last_eta=0, mode='linear'):
        super().__init__()
        if mode == 'linear':
            self.eta_scheduler = LinearScheduler(first_eta, last_eta, epochs)
        elif mode == 'cosine':
            self.eta_scheduler = CosineScheduler(first_eta, last_eta, epochs)

    def on_train_epoch_start(self, trainer, model):
        if model.model.pooler in ['baypool']:
            # Log and update coefficients
            eta = self.eta_scheduler(trainer.current_epoch)  # Calculate coefficients
            model.model.pool.eta = eta          # Update coefficients in model
            model.log('eta', eta)               # Log coefficients