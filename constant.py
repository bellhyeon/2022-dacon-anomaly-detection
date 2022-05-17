import pandas as pd

TRAIN_IMAGE_PATH = "../../../RaidData/dacon-anomaly-dataset/train"
TEST_IMAGE_PATH = "../../../RaidData/dacon-anomaly-dataset/test"

TRAIN_CSV_PATH = "../../../RaidData/dacon-anomaly-dataset/train_df.csv"
TEST_CSV_PATH = "../../../RaidData/dacon-anomaly-dataset/test_df.csv"
SUBMISSION_CSV_PATH = "../../../RaidData/dacon-anomaly-dataset/sample_submission.csv"

train_csv = pd.read_csv(TRAIN_CSV_PATH)

train_labels = train_csv["label"]
label_unique = {key: value for value, key in enumerate(train_labels.unique())}

LABEL_DICT = label_unique
LABEL_DECODE_DICT = {v: k for k, v in LABEL_DICT.items()}
LABELS = [label_unique[k] for k in train_labels]

SAVE_LOSS_BASED_MODEL_NAME = "loss_best.pt"
SAVE_MODEL_NAME = "model.pt"
SAVE_F1_BASED_MODEL_NAME = "f1_best.pt"