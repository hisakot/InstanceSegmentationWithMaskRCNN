import argparse

import numpy as np
import torch

import json
import glob

import common
import utils
from dataset import Dataset, CocoDataset, CocoDatasetDA
from trainer import trainer, validater
import transforms as T
import timm.optim.optim_factory as optim_factory
from engine import evaluate

def main(args):
    # model and device
    model = common.get_model_instance_segmentation(common.NUM_CLASSES, common.BACKBONE_MODEL)
    model, device = common.setup_device(model)
    if args.load_model != None:
        print(common.SAVE_MODEL_DIR + args.load_model)
        model.load_state_dict(torch.load(common.SAVE_MODEL_DIR + args.load_model))

    # dataset
    transforms = T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.RandomHorizontalFlip(),
            T.RandomPhotometricDistort(),
            T.ScaleJitter((224, 224)),
            T.FixedSizeCrop((224, 224))
            ])
    valid_transforms = T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32)
            ])
    train = CocoDatasetDA(root=common.TRAIN_ROOT,
            annFile=common.TRAIN_ANN,
            transforms=transforms,
            is_train=True)
    test = CocoDatasetDA(root=common.TEST_ROOT,
           annFile=common.TEST_ANN,
           transforms=valid_transforms,
           is_train=False)
#         train = CocoDatasetDA(root='../Dataset/EndoVis2017/instrument_1_4_training',
#                 annFile='../Dataset/EndoVis2017/instrument_1_4_training/COCO_ann.json',
#                 transforms=transforms, is_train=True)
#         test = CocoDatasetDA(root='../Dataset/EndoVis2017/instrument_5_8_training',
#                 annFile='../Dataset/EndoVis2017/instrument_5_8_training/COCO_ann.json',
#                 transforms=valid_transforms, is_train=False)
    train_size = int(len(train) * (8./10.))
    valid_size = len(train) - train_size
    train, valid = torch.utils.data.random_split(train, [train_size, valid_size])
    valid.transforms = valid_transforms

    trainloader = torch.utils.data.DataLoader(train, batch_size=common.BATCH_SIZE,
            shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    validloader = torch.utils.data.DataLoader(valid, batch_size=common.BATCH_SIZE,
            shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    testloader = torch.utils.data.DataLoader(test, batch_size=1,
            shuffle=False, num_workers=1, collate_fn=utils.collate_fn)

    # optimizer
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=common.LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.1)

    # wandb
    if args.wandb == True:
        import wandb
        config = {
                'backbone': common.BACKBONE_MODEL,
                'dataset': common.ITEM,
                'num_classes': common.NUM_CLASSES,
                'epochs': common.NUM_EPOCHS,
                'batchsize': common.BATCH_SIZE,
                'lr': common.LEARNING_RATE
                }
        wandb.init(
                project="Mask R-CNN",
                group=str(common.BACKBONE_MODEL),
                config=config,
                )

    # Training
    early_stopping = [np.inf, args.early_stopping, 0]
    for epoch in range(common.NUM_EPOCHS):
        print(f'-------- epoch {epoch} --------')
        # train
        train_loss = trainer(trainloader, model, device, optimizer, epoch)
        # validate
        with torch.no_grad():
            valid_loss = validater(validloader, model, device)
            if epoch % 10 == 0:
                model.eval()
                evaluate(model, testloader, device)
                # if you want to save model per 10 epoch
                #torch.save(model.state_dict(), common.SAVE_MODEL_DIR + str(epoch + 1))
        if args.wandb == True:
            wandb.log(
                    {
                     "Train Loss": train_loss,
                     "Valid Loss": valid_loss
                    }
            )

        # early stopping
        if valid_loss < early_stopping[0]:
            early_stopping[0] = valid_loss
            early_stopping[-1] = 0
            torch.save(model.state_dict(), common.SAVE_MODEL_DIR + str(epoch + 1))
            print(early_stopping)
        else:
            early_stopping[-1] += 1
            if early_stopping[-1] == early_stopping[1]:
                break



if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--early_stopping', type=int, default=50)
    args = parser.parse_args()
    main(args)
