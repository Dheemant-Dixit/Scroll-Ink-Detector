import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
import matplotlib.pyplot as plt
import sys
import os
import gc
import sys
import pickle
import warnings
import math
import time
import random
import argparse
import importlib
from tqdm.auto import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import segmentation_models_pytorch as smp
from warmup_scheduler import GradualWarmupScheduler
from segmentation_models_pytorch.encoders.mix_transformer import MixVisionTransformerEncoder
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import datetime

from test_config import CFG
from test_utils import *

IS_DEBUG = False
mode = 'train' if IS_DEBUG else 'test'
# top_TH = 0.5
# bot_TH = 0.4
# min_ar = 200
TH=0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if mode == 'test':
    fragment_ids = sorted(os.listdir(CFG.comp_dataset_path + mode))
else:
    fragment_ids = [1]

model = build_ensemble_model(device)

results = []
for fragment_id in fragment_ids:
    
    test_loader, xyxys = make_test_dataset(fragment_id)
    
    binary_mask = cv2.imread(CFG.comp_dataset_path + f"{mode}/{fragment_id}/mask.png", 0)
    binary_mask = (binary_mask / 255).astype(int)
    
    ori_h = binary_mask.shape[0]
    ori_w = binary_mask.shape[1]
    # mask = mask / 255

    pad0 = (CFG.tile_size - binary_mask.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - binary_mask.shape[1] % CFG.tile_size)

    binary_mask = np.pad(binary_mask, [(0, pad0), (0, pad1)], constant_values=0)
    
    mask_pred = np.zeros(binary_mask.shape)
    mask_count = np.zeros(binary_mask.shape)

    for step, (images) in tqdm(enumerate(test_loader), total=len(test_loader)):
#         print(images.shape)
        images = images.to(device)
        batch_size = images.size(0)

        with torch.no_grad():
            y_preds = TTA(images, model, device)
#             print(y_preds.shape)

        start_idx = step*CFG.batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0).cpu().numpy()
            mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))
    
    plt.imshow(mask_count)
    plt.show()
    
    print(f'mask_count_min: {mask_count.min()}')
    mask_pred /= (mask_count)
    
    del y_preds, mask_count
    gc.collect()
    torch.cuda.empty_cache()
    
    mask_pred=xp.array(mask_pred)
    mask_pred=denoise_image(mask_pred, iter_num=250)
    mask_pred=mask_pred.get()
    
    mask_pred = mask_pred[:ori_h, :ori_w]
    binary_mask = binary_mask[:ori_h, :ori_w]
    
    mask_pred = (mask_pred >= TH).astype(np.uint8)
    mask_pred = mask_pred.astype(int)
    mask_pred *= binary_mask
    
#     print(mask_pred.shape)
    
#     classification_mask = (mask_pred > top_TH).astype(int)
#     mask = mask_pred.copy()
#     mask[classification_mask.sum(axis=(1)) < min_ar, :] = np.zeros_like(mask_pred[0])
#     mask = (mask > bot_TH).astype(int)
    
    plt.imshow(mask_pred)
    plt.show()
    
    inklabels_rle = rle(mask_pred)
    
    results.append((fragment_id, inklabels_rle))
    

    del mask_pred
    del test_loader
    
    gc.collect()
    torch.cuda.empty_cache()

sub = pd.DataFrame(results, columns=['Id', 'Predicted'])

sample_sub = pd.read_csv(CFG.comp_dataset_path + 'sample_submission.csv')
sample_sub = pd.merge(sample_sub[['Id']], sub, on='Id', how='left')

sample_sub.to_csv("submission.csv", index=False)