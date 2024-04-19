import cv2
import numpy as np
import random
import torch
import common
import os
from PIL import Image
import torchvision
from torchvision.datasets import CocoDetection
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

def horizontal_flip(img, masks, boxes, p):
    if random.random() < p:
        img = img[:,::-1,:]
        if type(masks) is np.ndarray:
            mask_num = masks.shape[0]
        elif type(masks) is list:
            mask_num = len(masks)
        for idx in range(mask_num):
            masks[idx] = masks[idx][:,::-1]
        boxes = np.array(boxes)
        boxes[:, [0, 2]] = img.shape[1] - boxes[:, [2, 0]]
        boxes = boxes.tolist()

    return img, masks, boxes

def random_crop(img, masks, boxes, p):
    max_buf_x = int(img.shape[1] / 4)
    max_buf_y = int(img.shape[0] / 4)
    cropped_w = img.shape[1] - max_buf_x
    cropped_h = img.shape[0] - max_buf_y
# print("w:", img.shape[1], max_buf_x, cropped_w, "h:", img.shape[0], max_buf_y, cropped_h)
    if random.random() < p:
        buf_x = random.randint(0, max_buf_x)
        buf_y = random.randint(0, max_buf_y)
        if type(masks) is np.ndarray:
            mask_num = masks.shape[0]
        elif type(masks) is list:
            mask_num = len(masks)
        crop_masks = np.zeros((mask_num, cropped_h, cropped_w), dtype=np.uint8)
        for idx in range(mask_num):
            distance_x = abs((boxes[idx][0] + boxes[idx][2]) / 2 - (buf_x + cropped_w / 2))
            distance_y = abs((boxes[idx][1] + boxes[idx][3]) / 2 - (buf_y + cropped_h / 2))
            size_x = (boxes[idx][2] - boxes[idx][0]) / 2 + cropped_w / 2
            size_y = (boxes[idx][3] - boxes[idx][1]) / 2 + cropped_h / 2
            if distance_x < size_x and distance_y < size_y:
	        # img
                crop_img = img[buf_y:buf_y+cropped_h, buf_x:buf_x+cropped_w, :]
                img = cv2.resize(crop_img, (img.shape[1], img.shape[0]))
	        # masks
                crop_masks[idx] = masks[idx][buf_y:buf_y+cropped_h, buf_x:buf_x+cropped_w]
                masks[idx] = cv2.resize(crop_masks[idx], (img.shape[1], img.shape[0]))
                # boxes
                boxes[idx][0] = max(buf_x, boxes[idx][0])
                boxes[idx][1] = max(buf_y, boxes[idx][1])
                boxes[idx][2] = min(buf_x+cropped_w, boxes[idx][2])
                boxes[idx][3] = min(buf_y+cropped_h, boxes[idx][3])
    return img, masks, boxes

def illuminate(img, p):
    if random.random() < p:
        alpha = random.uniform(0.5, 1.5)
        beta = random.uniform(-50, 50)
        img = alpha * img + beta
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

class Dataset(object):
    def __init__(self, img_paths, annotation, is_train):
        self.img_paths = img_paths
        tmp = {}
        for k, v in annotation.items():
            tmp[v["filename"]] = v
        self.annotation = tmp
        self.is_train = is_train

    def __getitem__(self, idx):
        # load img
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path, 1)

        # load annotation
        regions = self.annotation[img_path.split(os.sep)[-1]]["regions"]
        num_objs = len(regions)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        masks = np.zeros((num_objs, img.shape[0], img.shape[1]), dtype=np.uint8)
        labels = np.zeros(num_objs, dtype=np.int64)
        area = np.zeros((num_objs, ), dtype=np.float32)

        for idx, region in enumerate(regions):
            tmp = region['shape_attributes']
            xs = tmp['all_points_x']
            ys = tmp['all_points_y']
            # bbox
            boxes[idx] = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
            # mask
            vertex = np.array([[x, y] for x, y in zip(xs, ys)])
            cv2.fillPoly(masks[idx], [vertex], 1)
            area[idx] = masks[idx].sum()
            # label
            labels[idx] = list(region['region_attributes'][common.ITEM].keys())[0]

        # data augmentation 21/12
        if self.is_train:
            img, masks, boxes = horizontal_flip(img, masks, boxes, p=0.5)
            img, masks, boxes = random_crop(img, masks, boxes, p=0.5)
            img = illuminate(img, p=0.5)
            if random.random() < 0.5:
                img = cv2.GaussianBlur(img, (5, 5), 0)

        img = img / 255.
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img.astype(np.float32))
        boxes = torch.from_numpy(boxes)
        target = {
            "boxes": boxes,
            "masks": torch.from_numpy(masks),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor(area),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64)
        }

        return img, target

    def __len__(self):
        return len(self.img_paths)


class CocoDatasetDA(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, is_train=False):
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.is_train = is_train

    def __getitem__(self, index):
        idx = self.ids[index]
        image = self._load_image(idx)
        target = self._load_target(idx)
        if len(target) == 0:
            print(idx)

        labels = [x['category_id'] for x in target]
        image_id = idx
        area = [x['area'] for x in target]
        iscrowd = [x['iscrowd'] for x in target]
        xywh_boxes = [x['bbox'] for x in target]
        boxes = list()
        for i, box in enumerate(xywh_boxes):
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            boxes.append([box[0], box[1], box[2], box[3]])
        masks = [self.coco.annToMask(x) for x in target]
        if self.is_train:
            image = np.array(image) # NOTE PIL to numpy
            image, masks, boxes = horizontal_flip(image, masks, boxes, p=0.5)
            image, masks, boxes = random_crop(image, masks, boxes, p=0.5)
            image = illuminate(image, p=0.5)
            if random.random() < 0.5:
                image = cv2.GaussianBlur(image, (5, 5), 0)
            image = Image.fromarray(image) # NOTE numpy to PIL

        targets = {}
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
# targets["boxes"] = torchvision.ops.box_convert(boxes,'xywh','xyxy')
        targets["boxes"] = boxes
        targets["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        targets["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        targets["image_id"] = torch.tensor([image_id])
        targets["area"] = torch.tensor(area)
        targets["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            image, targets = self.transforms(image, targets)

        return image, targets

class CocoDataset(CocoDetection):
    def __getitem__(self, index):
        idx = self.ids[index]
        image = self._load_image(idx)
        target = self._load_target(idx)

        boxes = [x['bbox'] for x in target]
        labels = [x['category_id'] for x in target]
        image_id = idx
        area = [x['area'] for x in target]
        iscrowd = [x['iscrowd'] for x in target]
        masks = [self.coco.annToMask(x) for x in target]

        targets = {}
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        targets["boxes"] = torchvision.ops.box_convert(boxes,'xywh','xyxy')
        targets["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        targets["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        targets["image_id"] = torch.tensor([image_id])
        targets["area"] = torch.tensor(area)
        targets["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            image, targets = self.transforms(image, targets)

        return image, targets
