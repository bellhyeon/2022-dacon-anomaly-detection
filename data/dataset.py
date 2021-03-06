import torch
import cv2
from typing import List

from albumentations import (
    Compose,
    Normalize,
    Resize,
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    Transpose,
)

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def _get_transforms(use_augmentation: bool, img_size: int):
    if use_augmentation:
        return Compose(
            [
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(rotate_limit=(-30, 30), p=0.5, border_mode=cv2.BORDER_CONSTANT),
                Resize(img_size, img_size),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    return Compose(
        [
            Resize(img_size, img_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


class AnomalyDataset(Dataset):
    def __init__(
        self,
        img_paths: List,
        labels: List,
        training: bool = True,
        img_size: int = 224,
        use_augmentation: bool = True,
    ):
        self.img_paths = img_paths
        self.labels = labels
        self.training = training
        self.img_size = img_size
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img = cv2.imread(self.img_paths[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = _get_transforms(self.use_augmentation, self.img_size)(image=img)
        img = augmented["image"]

        if self.training:
            label = self.labels[item]

            return {
                "input": img,
                "target": torch.tensor(label, dtype=torch.long),
            }

        return {
            "input": img,
        }
