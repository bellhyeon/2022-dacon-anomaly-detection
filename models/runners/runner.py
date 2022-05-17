from typing import Optional
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
from torch import nn


class Runner(metaclass=ABCMeta):
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        scheduler,
        loss_func,
        device,
        max_grad_norm,
    ):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._loss_func = loss_func
        self._device = device
        self._max_grad_norm = max_grad_norm

    @abstractmethod
    def forward(self, item):
        pass

    @abstractmethod
    def run(
        self,
        data_loader: DataLoader,
        epoch: Optional[int],
        training: bool,
        mixup: bool,
        cutmix: bool,
    ):
        pass
