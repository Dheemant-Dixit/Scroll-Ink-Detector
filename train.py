import numpy as np
import matplotlib.pyplot as plt
import gc
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
import cv2

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from segmentation_models_pytorch.encoders.mix_transformer import MixVisionTransformerEncoder
from train_config import CFG
from train_utils import *

cfg_init(CFG)

Logger = init_logger(log_file=CFG.log_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for fold in CFG.valid_id:
    
    train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset(fold)
    valid_xyxys = np.stack(valid_xyxys)
    
    
    train_dataset = CustomDataset(train_images, CFG, labels=train_masks,
                                  transform=get_transforms(data='train', cfg=CFG))

    valid_dataset = CustomDataset(valid_images, CFG, labels=valid_masks,
                                  transform=get_transforms(data='valid', cfg=CFG))

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.train_batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                              )

    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    
    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/{fold}/inklabels.png", 0)
    valid_mask_gt = valid_mask_gt / 255
    pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
    
    
    if CFG.metric_direction == 'minimize':
        best_score = np.inf
    elif CFG.metric_direction == 'maximize':
        best_score = -1

    best_loss = np.inf
    
    model = build_model(CFG)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    scheduler = get_scheduler(CFG, optimizer)

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, device)

        # eval
        avg_val_loss, mask_pred = valid_fn(
            valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt)

        scheduler_step(scheduler, avg_val_loss, epoch)

        best_dice, best_th = calc_cv(valid_mask_gt, mask_pred, Logger)

        # score = avg_val_loss
        score = best_dice

        elapsed = time.time() - start_time

        Logger.info(
            f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        Logger.info(
            f'Epoch {epoch+1} - avgScore: {score:.4f}')

        if CFG.metric_direction == 'minimize':
            update_best = score < best_score
        elif CFG.metric_direction == 'maximize':
            update_best = score > best_score

        if update_best:
            best_loss = avg_val_loss
            best_score = score

            Logger.info(
                f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            Logger.info(
                f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')

            torch.save({'model': model.state_dict()},
                        CFG.model_dir + f'{CFG.model_name}_fold{fold}_best.pth')
        torch.cuda.empty_cache()
        gc.collect()
    del model
    gc.collect()
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()