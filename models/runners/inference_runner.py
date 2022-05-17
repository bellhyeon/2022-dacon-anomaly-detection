from typing import List
from torch.utils.data import DataLoader
import torch
import numpy as np

import ttach as tta
from tqdm import tqdm
from models.runners.runner import Runner


class InferenceRunner(Runner):
    def forward(self, item):
        self._optimizer.zero_grad()
        inp = item["input"].to(self._device)
        output = self._model.forward(inp)
        return output

    def run(
        self,
        data_loader: DataLoader,
        epochs=None,
        training=False,
        mixup=False,
        cutmix=False,
    ):
        print("=" * 25 + "Start Inference" + "=" * 25)
        prediction: List = []
        assert not training
        assert not mixup
        assert not cutmix

        self._model.eval()
        with torch.no_grad():
            for item in tqdm(data_loader):
                output = self.forward(item)
                prediction.extend(torch.argmax(output, dim=-1).data.cpu().numpy())

        return prediction

    def infer(self, data_loader: DataLoader, training=False):
        print("=" * 25 + "Start Inference" + "=" * 25)
        prediction: List = []
        assert not training

        self._model.eval()
        with torch.no_grad():
            for item in tqdm(data_loader):
                output = self.forward(item)
                prediction.extend(output.data.cpu().numpy())
        prediction = np.array(prediction)

        return prediction

    def load_model(self, load_path):
        status = torch.load(load_path, map_location=torch.device("cpu"))
        self._model.load_state_dict(status["model"])
        self._optimizer.load_state_dict(status["optimizer"])
        self._scheduler.load_state_dict(status["scheduler"])
        self._tta_transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
            ]
        )
        self._model = tta.ClassificationTTAWrapper(self._model, self._tta_transforms)
        self._model.to(self._device)
