import argparse

import cv2
import numpy as np
import torch

import json
import gc
import glob
import os

import common
import utils
from dataset import Dataset, CocoDataset, CocoDatasetDA
from trainer import trainer, validater
import transforms as T
import timm.optim.optim_factory as optim_factory
from engine import evaluate
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

def test_cv(fold, test_dataset, model):
    confidense = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    for num, batch in tqdm(enumerate(test_dataset)):
        images, targets = batch
        img = images.detach().cpu().numpy() * 255
        img = img.transpose(1, 2, 0) # (3, 224, 224) -> (224, 224, 3)

        # target masks
        target = targets
        masks = (target["masks"]>confidense).squeeze().detach().cpu().numpy()
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]
        boxes = list(target["boxes"].detach().cpu().numpy())
        idxs = [range(len(targets["masks"]))]
        pred_cls = [common.CLASS_NAMES[label] for label in list(target["labels"].detach().cpu().numpy())]
        img = np.zeros_like(img)
        binary_mask = np.zeros((img.shape[0], img.shape[1], common.NUM_CLASSES))
        
        for i in range(len(masks)):
            mask = masks[i]
            binary_mask[:, :, common.CLASS_NAMES.index(pred_cls[i])] += mask

        binary_mask = binary_mask[:, :, 1:] # remove backgound rayer
        binary_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[0])) * 255
        np.clip(binary_mask, 0, 255)
        p = os.path.join("output", "g_t", "binary_mask", str(fold))
        if not os.path.isdir(p):
            os.makedirs(p)
        np.savez_compressed(os.path.join(p, f"{num}.npz"), binary_mask)
        binary_mask = binary_mask.sum(axis=2)
        cv2.imwrite(os.path.join(p, f"{num}.png"), binary_mask)

        # pred masks
        img /= 255.
        img = img.transpose(2, 0, 1) # (3, 224, 224)
        img = torch.from_numpy(img.astype(np.float32))

        pred = model([img.to(device)])

        pred_scores = pred[0]["scores"].detach().cpu().numpy()
        pred_scores_list = list(pred_scores)
        pred_idx = [idx for idx, score in enumerate(pred_scores) if score > confidense]
        masks = (pred[0]["masks"]>0.5).squeeze().detach().cpu().numpy()
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]
        pred_class = [common.CLASS_NAMES[i] for i in list(pred[0]["labels"].cpu().numpy())]

        if len(pred_idx) == 0:
            scores = []
            masks = []
            pred_cls = []
        else:
            scores = [pred_scores[idx] for idx in pred_idx]
            masks = [masks[idx] for idx in pred_idx]
            pred_cls = [pred_class[idx] for idx in pred_idx]

        img = np.zeros_like(img)
        binary_mask = np.zeros((img.shape[1], img.shape[2], common.NUM_CLASSES)) # (224, 224, 9)
        
        for i in range(len(scores)):
            mask = masks[i]
            binary_mask[:, :, common.CLASS_NAMES.index(pred_cls[i])] += mask
        binary_mask = binary_mask[:, :, 1:]
        binary_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[2])) * 255
        np.clip(binary_mask, 0, 255)
        p = os.path.join("output", "pred", "binary_mask", str(fold))
        if not os.path.isdir(p):
            os.makedirs(p)
        np.savez_compressed(os.path.join(p, f"{num}.npz"), binary_mask)
        binary_mask = binary_mask.sum(axis=2)
        cv2.imwrite(os.path.join(p, f"{num}.png"), binary_mask)

def main(args):
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

#         train_size = int(len(train) * (8./10.))
#         valid_size = len(train) - train_size
#         train, valid = torch.utils.data.random_split(train, [train_size, valid_size])
#         valid.transforms = valid_transforms
#     trainloader = torch.utils.data.DataLoader(train, batch_size=common.BATCH_SIZE,
#             shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
#     validloader = torch.utils.data.DataLoader(valid, batch_size=common.BATCH_SIZE,
#             shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
#     testloader = torch.utils.data.DataLoader(test, batch_size=1,
#             shuffle=False, num_workers=1, collate_fn=utils.collate_fn)

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
    folds = KFold(n_splits=common.FOLD_NUM, shuffle=True, random_state=common.SEED)
    train_loss_list = []
    valid_loss_list = []

    for fold, (trainval_idx, test_idx) in enumerate(folds.split(train)):
        # model and device
        model = common.get_model_instance_segmentation(common.NUM_CLASSES,
                                                       common.BACKBONE_MODEL)
        model, device = common.setup_device(model)
        if args.load_model != None:
            logger.info(common.SAVE_MODEL_DIR + args.load_model)
#print(common.SAVE_MODEL_DIR + args.load_model)
            model.load_state_dict(torch.load(common.SAVE_MODEL_DIR + args.load_model))

        # optimizer
        param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
        optimizer = torch.optim.AdamW(param_groups, lr=common.LEARNING_RATE,
                                      betas=(0.9, 0.999), weight_decay=0.1)

        # load dataset
        trainval_size = trainval_idx.size
        train_size = int(trainval_size * 0.8)
        val_size = trainval_size - train_size

        train_dataset = torch.utils.data.dataset.Subset(train, trainval_idx[:train_size])
        valid_dataset = torch.utils.data.dataset.Subset(train, trainval_idx[train_size:])
        test_dataset = torch.utils.data.dataset.Subset(train, test_idx)

        trainloader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size=common.BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=4,
                                                  collate_fn=utils.collate_fn)
        validloader = torch.utils.data.DataLoader(valid_dataset,
                                                  batch_size=common.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  collate_fn=utils.collate_fn)
        testloader = torch.utils.data.DataLoader(test_dataset, 
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=1,
                                                 collate_fn=utils.collate_fn)

        # for visualization
        train_losses = []
        valid_losses = []
        early_stopping = [np.inf, args.early_stopping, 0]
        print(early_stopping)
        # start training
        for epoch in range(common.NUM_EPOCHS):
            print(f'-------- {fold} fold: epoch {epoch} --------')
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

                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
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
                if os.path.isdir(common.SAVE_MODEL_DIR + str(fold)):
                    pass
                else:
                    os.mkdir(common.SAVE_MODEL_DIR + str(fold))
                torch.save(model.state_dict(), common.SAVE_MODEL_DIR + str(fold) + "/" + str(epoch + 1))
                print(early_stopping)
            else:
                early_stopping[-1] += 1
                print(early_stopping)
                if early_stopping[-1] == early_stopping[1]:
                    train_loss_list.append(train_losses)
                    valid_loss_list.append(valid_losses)
                    with torch.no_grad():
                        test_cv(fold, test_dataset, model)
                    del model, optimizer, trainloader, validloader, testloader, train_losses, valid_losses
                    gc.collect()
                    torch.cuda.empty_cache()
                    break

    common.show_validation_score(train_loss_list, valid_loss_list, save=True)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--early_stopping', type=int, default=50)
    args = parser.parse_args()
    main(args)
