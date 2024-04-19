
import torch
import torch.nn as nn
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from backbones import ResNetBackBoneWithFPN, ViTBackboneWithFPN
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import matplotlib.pyplot as plt
import math
import numpy as np
import os

ITEM = 'tool'
CLASS_NAMES = ['background', 'forceps', 'tweezers', 'electrical-scalpel',
            'scalpel', 'hook', 'syringe', 'needle-holder', 'pen']

colors = [[0, 0, 0],[255, 0, 255],[0, 255, 255],[255, 255, 0],[80, 70, 180],[180, 40, 250],[245, 145, 50],[70, 150, 250],[50, 190, 190]]

# CLASS_NAMES = ['background', 'pen', 'syringe', 'scalpel', 'e_scalpel',
# 	       'e_scalpel_scissors', 'forceps', 'needleholder', 'hook', 'tweezers']
# colors = [[0, 0, 0],[0, 255, 0],[0, 255, 255],[255, 255, 0],[80, 70, 180],[180, 40, 250],[245, 145, 50],[70, 150, 250],[50, 190, 190], [255, 0, 0]]

BACKBONE_MODEL = 'resnet'

NUM_CLASSES = len(CLASS_NAMES)
num_coco_classes = 92
NUM_EPOCHS = 1000
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MIN_LERNING_RATE = 1e-8
# WARMUP_EPOCHS = 40
WARMUP_EPOCHS = 20
SAVE_MODEL_DIR ="../models/"

# for Cross Validation
FOLD_NUM = 2
SEED = 0
GRAPH_SAVE_DIR = "./cv_loss_graphs/"

# for main.py
TRAIN_ROOT = "../data/ss_2nd_step_train"
TRAIN_ANN = "../data/ss_2nd_step_train/COCO_train_annos_re.json"
TEST_ROOT = "../data/val"
TEST_ANN = "../data/val/COCO_val_annos.json"

ANNOTATION_FILE = "../data/annotation_data_multi_tools3.json"

# TEST_IMG_PATH = "./breast_surgery3/val/*.png"
# TEST_IMG_PATH = "F:/Dataset/org_imgs/2/*.png" # for 1st inference
# TEST_IMG_PATH = "original_20200214_1_images/*.png" # for 1st inference
# TEST_IMG_PATH = "../Dataset/assessment_tools/org_imgs/*.png" # for 2nd inference
TEST_IMG_PATH = "../../Dataset/surgery/org_imgs/6/*.png"

SAVE_COLOR_DIR = "./output/multitools_color/"
SAVE_BINARY_DIR = "./output/multitools_binary/"

IMG_W = 960
IMG_H = 540

def setup_device(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use GPU ----------")
    else:
        print("---------- Use CPU ----------")
    model.to(device)

    return model, device

def get_model_instance_segmentation(num_classes, backbone_model=BACKBONE_MODEL):
    if backbone_model == "vit" or backbone_model == 'resnet':
        # reducing anchorboxes may have positive effects on result if model predicts too many bbox on test image
        # sizes are relative to images resolution
        # vit use (224, 224) images, so anchor box sizes are [0-224]
        anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 196), (16, 32, 64, 128, 196), (16, 32, 64, 128, 196), (16, 32, 64, 128, 196)),
                aspect_ratios=((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)))
    else:
        pass

    if backbone_model ==  "resnet":
        backbone = ResNetBackBoneWithFPN()
    elif backbone_model == "vit":
        backbone = ViTBackboneWithFPN()
    else:
        pass
    print(f'-------BackBone is {backbone_model}-------')
    model = MaskRCNN(backbone,
                    num_classes=num_classes,
# fixed_size=(224, 224), # Note: version niyottekawatta?
                    rpn_anchor_generator=anchor_generator)
    return model

def show_validation_score(train_loss_list, valid_loss_list, save=False, save_dir=GRAPH_SAVE_DIR):
    fig = plt.figure(figsize=(15, 15))
    for i in range(FOLD_NUM):
        train_loss = train_loss_list[i]
        valid_loss = valid_loss_list[i]

        ax = fig.add_subplot(math.ceil(np.sqrt(FOLD_NUM))*2, math.ceil(np.sqrt(FOLD_NUM))*2, (i*2)+1, title=f'Fold {i+1}')
        ax.plot(range(len(train_loss)), train_loss, c="orange", label="train")
        ax.plot(range(len(valid_loss)), valid_loss, c="blue", label="valid")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()

    plt.tight_layout()
    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + "loss_graphs.png")
    else:
        plt.show()
