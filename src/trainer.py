import argparse, os, glob, sys, json, random, tqdm
import gc
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
import math

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# from torch.utils.tensorboard import SummaryWriter

sys.path.append('../vision/references/detection'.replace('/', os.sep))
import engine, utils
import common
from dataset import Dataset
import transforms as T

copypaste = T.SimpleCopyPaste()

def trainer(trainloader, model, device, optimizer, epoch):
    print("---------- Start Training ----------")
    model.train()
    try:
        with tqdm(enumerate(trainloader), total=len(trainloader), ncols=100) as pbar:
            train_loss = 0.0
            for i, batch in pbar:
                utils.adjust_learning_rate(optimizer, i/len(trainloader)+epoch,
                        common.NUM_EPOCHS,
                        common.LEARNING_RATE,
                        common.MIN_LERNING_RATE,
                        common.WARMUP_EPOCHS)
                images, targets = copypaste(*batch)
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                del losses
                gc.collect()

                train_loss += loss_value
        return train_loss
    except ValueError:
        pass

def validater(validloader, model, device):
    print("---------- Start Validating ----------")
    model.train()
    try:
        with tqdm(enumerate(validloader), total=len(validloader), ncols=100) as pbar:
            valid_loss = 0.0
            for i, batch in pbar:
                images, targets = batch
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                valid_loss += loss_value
        return valid_loss
    except ValueError:
        pass

if __name__ == '__main__':
    pass