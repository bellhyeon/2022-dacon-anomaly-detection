import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .runner import Runner
import torch.nn as nn

from utils.mixup import mixup_data, mixup_criterion
from utils.cutmix import cutmix_data

from constant import LABEL_DICT


def _save_loss_graph(save_folder_path: str, train_loss: List, valid_loss: List):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_loss, label="train_loss")
    plt.plot(valid_loss, label="valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss", fontsize=15)
    plt.legend()

    save_path = os.path.join(save_folder_path, "loss.png")

    plt.savefig(save_path)


def _save_acc_graph(
    save_folder_path: str,
    train_acc: List,
    valid_acc: List,
):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_acc, label="train_acc")
    plt.plot(valid_acc, label="valid_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Accuracy", fontsize=15)
    plt.legend()
    save_path = os.path.join(save_folder_path, "acc.png")
    plt.savefig(save_path)


def _save_f1_graph(save_folder_path: str, train_f1: List, valid_f1: List):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_f1, label="train_f1")
    plt.plot(valid_f1, label="valid_f1")
    plt.xlabel("epoch")
    plt.ylabel("f1")
    plt.title("F1 Score", fontsize=15)
    plt.legend()
    save_path = os.path.join(save_folder_path, "f1.png")

    plt.savefig(save_path)


def _calc_accuracy(prediction, label):
    prediction = torch.argmax(prediction, dim=-1).data.cpu().numpy()
    label = label.data.cpu().numpy()

    accuracy = accuracy_score(label, prediction)

    return accuracy


def _calc_f1(prediction, label):
    prediction = torch.argmax(prediction, dim=-1).data.cpu().numpy()
    label = label.data.cpu().numpy()

    f1 = f1_score(prediction, label, average="macro")
    return f1, prediction, label


class TrainingRunner(Runner):
    def __init__(
        self, model: nn.Module, optimizer, scheduler, loss_func, device, max_grad_norm
    ):
        super().__init__(model, optimizer, scheduler, loss_func, device, max_grad_norm)
        self._valid_predict: List = []
        self._valid_label: List = []

    def forward(self, item):
        inp = item["input"].to(self._device)
        target = item["target"].to(self._device)
        output = self._model.forward(inp.float())
        acc = _calc_accuracy(output, target)
        f1, prediction, label = _calc_f1(output, target)

        return self._loss_func(output, target), acc, f1, prediction, label

    def _mixup_forward(self, item):
        inp = item["input"].to(self._device)
        target = item["target"].to(self._device)

        inp, target_a, target_b, lam = mixup_data(inp, target, self._device)
        output = self._model.forward(inp)

        acc = _calc_accuracy(output, target)
        f1, prediction, label = _calc_f1(output, target)

        return (
            mixup_criterion(self._loss_func, output, target_a, target_b, lam),
            acc,
            f1,
            prediction,
            label,
        )

    def _cutmix_forward(self, item):
        inp = item["input"].to(self._device)
        target = item["target"].to(self._device)

        inp, target_a, target_b, lam = cutmix_data(inp, target, alpha=1.0)
        output = self._model.forward(inp)

        acc = _calc_accuracy(output, target)
        f1, prediction, label = _calc_f1(output, target)

        loss = lam * self._loss_func(output, target_a) + (1 - lam) * self._loss_func(
            output, target_b
        )
        return (
            loss,
            acc,
            f1,
            prediction,
            label,
        )

    def _backward(self, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()

    def run(
        self,
        data_loader: DataLoader,
        epoch: int,
        training=True,
        mixup=False,
        mixup_epochs=None,
        cutmix=False,
        cutmix_epochs=None,
    ):
        total_loss: float = 0.0
        total_acc: float = 0.0
        total_f1: float = 0.0
        total_batch: int = 0
        train_batch: int = 0
        if training:
            if mixup and epoch < mixup_epochs:
                print("=" * 25 + f"Epoch {epoch} Train with Mixup" + "=" * 25)
            elif cutmix and epoch < cutmix_epochs:
                print("=" * 25 + f"Epoch {epoch} Train with Cutmix" + "=" * 25)
            else:
                print("=" * 25 + f"Epoch {epoch} Train" + "=" * 25)

            self._model.train()
            for item in tqdm(data_loader):
                self._optimizer.zero_grad()

                if mixup and epoch < mixup_epochs:
                    loss, acc, f1, _, _ = self._mixup_forward(item)
                elif cutmix and epoch < cutmix_epochs:
                    loss, acc, f1, _, _ = self._cutmix_forward(item)
                else:
                    loss, acc, f1, _, _ = self.forward(item)

                total_loss += loss.item()
                total_acc += acc
                total_f1 += f1
                total_batch += 1
                train_batch += 1
                self._backward(loss)

            return (
                round((total_loss / total_batch), 4),
                round((total_acc / total_batch), 4),
                round((total_f1 / total_batch), 4),
            )

        else:
            print("=" * 25 + f"Epoch {epoch} Valid" + "=" * 25)
            self._model.eval()
            with torch.no_grad():
                for item in tqdm(data_loader):
                    loss, acc, f1, prediction, label = self.forward(item)
                    self._valid_predict.extend(prediction)
                    self._valid_label.extend(label)

                    total_loss += loss.item()
                    total_acc += acc
                    total_f1 += f1
                    total_batch += 1

        return (
            round((total_loss / total_batch), 4),
            round((total_acc / total_batch), 4),
            round((total_f1 / total_batch), 4),
        )

    def save_model(self, save_path):
        torch.save(
            {
                "model": self._model.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "scheduler": self._scheduler.state_dict(),
            },
            save_path,
        )

    @staticmethod
    def save_result(
        epoch: int,
        save_folder_path: str,
        train_loss: float,
        valid_loss: float,
        train_acc: float,
        valid_acc: float,
        train_f1_score: float,
        valid_f1_score: float,
        args,
        loss_based: bool = True,
    ):
        if epoch == 0:  # save only once
            save_json_path = os.path.join(save_folder_path, "model_spec.json")
            with open(save_json_path, "w") as json_file:
                save_json = args.__dict__
                json.dump(save_json, json_file)

        if loss_based:
            save_result_path = os.path.join(save_folder_path, "loss_best_result.json")
        else:
            save_result_path = os.path.join(save_folder_path, "f1_best_result.json")
        with open(save_result_path, "w") as json_file:
            save_result_dict: Dict = {
                "best_epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_f1_score": train_f1_score,
                "valid_f1_score": valid_f1_score,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
            }

            json.dump(save_result_dict, json_file)
        if loss_based:
            print("Save Best Loss Model and Graph")
        else:
            print("Save Best F1 Model and Graph")

    @staticmethod
    def save_graph(
        save_folder_path: str,
        train_loss: List,
        train_acc: List,
        train_f1_score: List,
        valid_loss: List,
        valid_acc: List,
        valid_f1_score: List,
    ):
        _save_loss_graph(save_folder_path, train_loss, valid_loss)
        _save_acc_graph(save_folder_path, train_acc, valid_acc)
        _save_f1_graph(save_folder_path, train_f1_score, valid_f1_score)

    def save_confusion_matrix(self, save_folder_path: str, loss_based: bool = True):
        matrix = confusion_matrix(self._valid_label, self._valid_predict)

        data_frame = pd.DataFrame(
            matrix, columns=list(LABEL_DICT.keys()), index=list(LABEL_DICT.keys())
        )

        plt.figure(figsize=(60, 60))
        sns.heatmap(
            data_frame,
            cmap="Blues",
            annot_kws={"size": 6},
            annot=True,
            linecolor="grey",
            linewidths=0.3,
        )
        plt.xticks(rotation=-90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Prediction")
        plt.ylabel("Answer")
        if loss_based:
            save_path = os.path.join(save_folder_path, "loss_best_confusion_matrix.png")
        else:
            save_path = os.path.join(save_folder_path, "f1_best_confusion_matrix.png")
        plt.savefig(save_path)
        print(f"matrix.shape: {matrix.shape}\n")
