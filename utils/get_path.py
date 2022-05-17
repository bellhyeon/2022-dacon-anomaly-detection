from glob import glob
import os
from constant import LABELS


def load_path(dataset_path, train=True):
    if train:
        img_paths = sorted(glob(dataset_path + "/*"))
        labels = LABELS

        return img_paths, labels
    else:
        img_paths = sorted(glob(dataset_path + "/*"))
        return img_paths


def get_save_kfold_model_path(save_path: str, save_model_name: str, fold_num: int):
    # fold 저장할 폴더
    save_folder_path = os.path.join(save_path, str(fold_num + 1))

    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    save_model_path = os.path.join(save_folder_path, save_model_name)
    print(f"Model Save Path : {save_folder_path}")

    return save_model_path, save_folder_path


def get_save_model_path(save_path: str, save_model_name: str):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    num_folder = len(glob(save_path + "*"))
    save_folder_path = os.path.join(save_path, str(num_folder + 1))

    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)
    save_model_path = os.path.join(save_folder_path, save_model_name)
    print(f"Model Save Path: {save_folder_path}")

    return save_model_path, save_folder_path
