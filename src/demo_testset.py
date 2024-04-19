import argparse
import csv
import cv2
import datetime
import glob
import numpy as np
import os

import torch
import torchvision
from tqdm import tqdm

import common
from dataset import Dataset, CocoDataset, CocoDatasetDA
import json
import utils
import transforms as T

def get_coloured_mask(mask, pred_cls, boxes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    colors = common.colors
    b[mask == 1], g[mask == 1], r[mask == 1] = colors[common.CLASS_NAMES.index(pred_cls)]
    coloured_mask = np.stack([b, g, r], axis=2)
    
    b, g, r = colors[common.CLASS_NAMES.index(pred_cls)]
#     cv2.rectangle(coloured_mask, (boxes[0]), (boxes[1]), (b, g, r), thickness=2)
#     cv2.putText(coloured_mask, text=pred_cls, org=boxes[0],
# 		fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
# 		color=(b, g, r), thickness=1, lineType=cv2.LINE_AA)
    return coloured_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--save_color_mask', required=False, action='store_true')
    parser.add_argument('--save_binary_mask', required=False, action='store_true')
    parser.add_argument('--save_coco_json', required=False, action='store_true')
    parser.add_argument('--output_dir', default='output')
    args = parser.parse_args()

    # model and device
    best_model_path = common.SAVE_MODEL_DIR + args.model_name
    model = common.get_model_instance_segmentation(common.NUM_CLASSES)
    model, device = common.setup_device(model)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    confidence = 0.4

    # Prediction for val dataset NOTE it is test set correctly
    transforms = T.Compose([
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float32)
        ])
#     dataset = CocoDataset(root='breast_surgery2/val',
#                                 annFile='breast_surgery2/val/COCO_val_annos.json',
#                                 transforms=transforms)
    dataset = CocoDatasetDA(root='breast_surgery2/val',
                            annFile='breast_surgery2/val/COCO_val_annos.json',
                            transforms=transforms, is_train=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
            shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # evaluate test set
    for idx, batch in tqdm(enumerate(dataset)):
        images, targets = batch
        img = images.detach().cpu().numpy() * 255
        img = img.transpose(1, 2, 0)
        target = targets
        masks = (target['masks']>confidence).squeeze().detach().cpu().numpy()
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]
        boxes = list(target['boxes'].detach().cpu().numpy())
        idxs = [range(len(targets['masks']))]
        pred_cls = [common.CLASS_NAMES[label] for label in  list(target['labels'].detach().cpu().numpy())]
        img = np.zeros_like(img)
        binary_mask = np.zeros((img.shape[0], img.shape[1], common.NUM_CLASSES))
        color_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for i in range(len(target['boxes'])):
            box = boxes[i].astype(np.int32)
            cls = pred_cls[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            c = tuple(common.colors[common.CLASS_NAMES.index(cls)])
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=c, thickness=2)
            font_size = cv2.getTextSize(cls, font, 0.5, 2)[0]
            cv2.rectangle(img, (box[0], box[1] - font_size[1] - 2),
                          (box[0] + font_size[0], box[1] - 2), c, -1)
            cv2.putText(img, cls, (box[0], box[1] - 2), font, 0.5, (0, 0, 0),
                        thickness=1, lineType=cv2.LINE_AA)

        p = os.path.join('output', 'g_t', 'bbox')
        if not os.path.isdir(p):
            os.makedirs(p)
        cv2.imwrite(os.path.join(p, f'{idx}.png'), img)

        for i in range(len(masks)):
            if args.save_color_mask:
                rgb_mask = get_coloured_mask(masks[i], pred_cls[i], boxes[i])
                rgb_mask = cv2.resize(rgb_mask, (img.shape[1], img.shape[0]))
                color_mask = cv2.addWeighted(color_mask, 1, rgb_mask, 1, 0)

            if args.save_binary_mask:
                mask = masks[i]
                binary_mask[:, :, common.CLASS_NAMES.index(pred_cls[i])] += mask

        if args.save_color_mask:
            color_mask = cv2.resize(color_mask, (img.shape[1], img.shape[0]))
            p = os.path.join('output', 'g_t', 'color_mask')
            if not os.path.isdir(p):
                os.makedirs(p)
            cv2.imwrite(os.path.join(p, f'{idx}.png'), color_mask)

        if args.save_binary_mask:    
            binary_mask = binary_mask[:, :, 1:] # remove background rayer
            binary_mask = binary_mask.sum(axis=2)
            binary_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[0])) * 255
            np.clip(binary_mask, 0, 255)
            p = os.path.join('output', 'g_t', 'binary_mask')
            if not os.path.isdir(p):
                os.makedirs(p)
            cv2.imwrite(os.path.join(p, f'{idx}.png'), binary_mask)
