import torch
import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=1.0):

    shuffle = torch.randperm(x.size(0))
    cutmix_x = x.clone()

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)

    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    cutmix_x[:, :, x1:x2, y1:y2] = x[shuffle, :, x1:x2, y1:y2]
    # Adjust lambda to match pixel ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[shuffle]
    return cutmix_x, y_a, y_b, lam
