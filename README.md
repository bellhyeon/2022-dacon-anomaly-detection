# Computer Vision 이상치 탐지 알고리즘 경진대회
불균형 데이터 셋을 학습하여 사물의 상태를 잘 분류할 수 있는 알고리즘 개발
<br>[Competition Link](https://dacon.io/competitions/official/235894/overview/description)
* 주최 / 주관: Dacon
* **Private 11th, Score 0.8830**
* **Final 9th (9/481, 2%)**
***

## Structure
Train/Test data folder and sample submission file must be placed under **dataset** folder.<br>
If you want change dataset path, you can change in CONSTANT.py
```
repo
  |——dataset
        |——train
                |——10000.png
                |——....
        |——test
                |——20000.png
                |——....
        |——train_df.csv
        |——sample_submission.csv
  |——models
        |——model
        |——runners
  |——data
  |——utils
```
***

## Development Environment
* Ubuntu 18.04.5
* i9-10900X
* RTX 3090 1EA
* CUDA 11.3
***

## Install Dependencies (GPU)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3812/)

```shell
sh install_dependency.sh
```
***

## Solution

### Train
* Fine-Tuned timm tf_efficientnet_b6
* Image Size (528x528)
* Focal Loss (alpha=0.25, gamma=5.0) with Label Smoothing (0.1)
* Trained for 70 epochs
* 5 StratifiedKFold train
* Train 30 epochs with mixup, trained remaining epochs without mixup
* Transpose, Resize, HorizontalFlip, VerticalFlip, ShiftScaleRotate(-30, 30), Normalize

```shell
python kfold_main.py
```
***

### Inference
5 fold ensemble (soft-voting) with Test Time Augmentation
* HorizontalFlip, VerticalFlip

```shell
python kfold_inference.py
```
***
## Tried Techniques
* CutMix
* ArcFace Loss
* Model Ensemble (EffNetB7 + EffNetB6)
