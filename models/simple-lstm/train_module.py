from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, F1Score


class SimpleTrainModule(LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-4, loss_fn: nn.Module = nn.NLLLoss()) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn
        self.acc = Accuracy()
        self.f1 = F1Score(num_classes=2)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(params=self.model.parameters(), lr=self.lr)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: tuple, is_train: bool = True) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        if is_train:
            return self.loss_fn(y_hat, y.long().squeeze(dim=-1))
        else:
            self.acc(y_hat, y.squeeze())
            self.f1(y_hat, y.squeeze())
            self.log('Accuracy', self.acc, on_step=False, on_epoch=True)
            self.log('F1', self.f1, on_step=False, on_epoch=True)
            return self.loss_fn(y_hat, y.long().squeeze(dim=-1))

    def training_step(self, batch, *args, **kwargs) -> dict:
        loss = self._step(batch=batch)
        self.log('loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, *args, **kwargs) -> Optional[dict]:
        loss = self._step(batch=batch, is_train=False)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return {'val_loss': loss}
