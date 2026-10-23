import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision import models

# data/ at the repo root holds the three split files made by 1_baseline/split_data.py
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('WAFER_DATA_DIR', REPO / 'data'))
TRAIN_PATH = DATA_DIR / 'dataset_70_TRAIN.npz'
VAL_PATH = DATA_DIR / 'dataset_15_VAL.npz'
TEST_PATH = DATA_DIR / 'dataset_15_TEST.npz'

# the teacher is the same trained ResNet-18 the web demo uses
RESNET_BCE_CKPT = REPO / 'dashboard' / 'resnet18_bce.pth'

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
NUM_CLASSES = 8
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def resnet18_bce():
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def load_split(path):
    d = np.load(path)
    return d['x'], d['y']


# same preprocessing the ResNet-18 was trained with: 3 channels, ImageNet normalisation
def paper_input(x_uint8):
    t = torch.from_numpy(x_uint8).float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
    return (t - MEAN) / STD


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


@torch.no_grad()
def predict_logits(model, x_uint8, to_input=paper_input, batch=256):
    dev = device()
    model.to(dev).eval()
    out = [model(to_input(x_uint8[i:i + batch]).to(dev)).float().cpu() for i in range(0, len(x_uint8), batch)]
    return torch.cat(out).numpy()


# a logit above 0 is the same as a sigmoid output above 0.5
def scores(y, logits, thresh=0.0):
    pred = (logits > thresh).astype(int)
    per = f1_score(y, pred, average=None, zero_division=0)
    return {'macro_f1': float(per.mean()),
            'micro_f1': float(f1_score(y, pred, average='micro', zero_division=0)),
            'per_class_f1': {c: float(f) for c, f in zip(CLASSES, per)}}
