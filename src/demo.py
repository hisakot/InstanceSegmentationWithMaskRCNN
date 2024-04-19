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

def get_binary_mask(mask, pred_cls):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    binary = [255, 255, 255]
    b[mask == 1], g[mask == 1], r[mask == 1] = binary
    binary_mask = np.stack([b, g, r], axis=2)

    return binary_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--save_bbox', required=False, action='store_true')
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

    # Prediction for common.TEST_IMG_PATH
    img_paths = glob.glob(common.TEST_IMG_PATH)
    for idx in tqdm(range(len(img_paths))):
        if idx == idx:
            img_path = img_paths[idx]
            img = cv2.imread(img_path) # (h, w, c)
            img = img / 255.
            img = img.transpose(2,0,1) # c, h, w)
            img = torch.from_numpy(img.astype(np.float32))

            pred = model([img.to(device)])

            pred_scores = pred[0]['scores'].detach().cpu().numpy()
            pred_scores_list = list(pred_scores)
            pred_idx = [idx for idx, score in enumerate(pred_scores) if score > confidence]
            masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()

            '''
            No object was predicted
            EX) if masks = w:960, h:540 -> c:1, w:960, h:540
            '''
            if masks.ndim == 2:
                masks = masks[np.newaxis, :, :]

            pred_class = [common.CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
            pred_boxes = list(pred[0]['boxes'].detach().cpu().numpy())
            if len(pred_idx) == 0:
                scores = []
                masks = []
                boxes = []
                pred_cls = []
            else:
                scores = [pred_scores[idx] for idx in pred_idx]
                masks = [masks[idx] for idx in pred_idx]
                boxes = [pred_boxes[idx] for idx in pred_idx]
                pred_cls = [pred_class[idx] for idx in pred_idx]

            # init mask
            img = cv2.imread(img_path)
            img = np.zeros_like(img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            binary_mask = np.zeros((img.shape[0], img.shape[1], common.NUM_CLASSES))
            color_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            bw_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

            '''
            To save bounding box
            blending original images
            '''
            if args.save_bbox:
                for i in range(len(scores)):
                    box = boxes[i].astype(np.int32)
                    cls = pred_cls[i]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    color = tuple(common.colors[common.CLASS_NAMES.index(cls)])
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                                  color=color, thickness=2)
                    font_size = cv2.getTextSize(cls, font, 0.5, 2)[0]
                    cv2.rectangle(img, (box[0], box[1] - font_size[1] - 2),
                                  (box[0] + font_size[0], box[1] - 2), color, -1)
                    cv2.putText(img, cls, (box[0], box[1] - 2), font, 0.5,
                                (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                p = os.path.join(args.output_dir, 'bbox')
                if not os.path.isdir(p):
                    os.makedirs(p)
                cv2.imwrite(os.path.join(p, img_paths[idx].split(os.sep)[-1]), img)

            '''
            To save color mask or binary mask or coco json file
            '''
            for i in range(len(scores)):
                coco_output = {}
                box = boxes[i]
                bbox = [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])]

                org_img = cv2.imread(img_path)
                bw_mask[:, :, 0] += masks[i]
                bw_mask *= 255
                cv2.imshow("bw", bw_mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
#                 blend_img = cv2.addWeighted(org_img, 1, bw_mask, 1, 0)
#                 cv2.rectangle(blend_img,
#                              (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
#                              color=(255, 0, 255), thickness=2)
#                 cv2.putText(blend_img, pred_cls[i], (30, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
#                             thickness=2, lineType=cv2.LINE_AA)
#                 cv2.imwrite("./output/blend_img/" + os.path.basename(img_path)[:-4] + "_" + str(i) + ".png", blend_img)

                contour, hierarchy = cv2.findContours(bw_mask[:, :, 0],
                                                      mode=cv2.RETR_LIST,
                                                      method=cv2.CHAIN_APPROX_SIMPLE)
                if len(contour) != 0:
                    area = cv2.contourArea(contour[0])
                else:
                    continue
                # save dict json for COCO Format dataset
                if args.save_coco_json:
                    img_dict = dict()
                    annotation_dict = dict()
                    img_dict = {"id" : 0,
                                "file_name" : os.path.basename(img_path),
                                "width" : img.shape[1],
                                "height" : img.shape[0],
                                "data_captured" : datetime.datetime.utcnow().isoformat(' '),
                                "coco_url" : "",
                                "flickr_url" : ""
                                }
                    annotations_dict = {"id" : 0,
                                        "image_id" : 0,
                                        "category_id" : common.CLASS_NAMES.index(pred_cls[i]),
                                        "iscrowd" : 0,
                                        "area" : area,
                                        "bbox" : bbox,
                                        "segmentation" : [contour[0].ravel().tolist()]
                                        }
                    coco_output["images"] = img_dict
                    coco_output["annotations"] = annotations_dict
                    with open("output/coco_json/" + os.path.basename(img_path) + str(i) + ".json",
                              'w', encoding="utf-8") as outfile:
                        json.dump(coco_output, outfile, sort_keys=True, indent=4, ensure_ascii=False)

                if args.save_color_mask:
                    rgb_mask = get_coloured_mask(masks[i], pred_cls[i], boxes[i])
                    rgb_mask = cv2.resize(rgb_mask, (img.shape[1], img.shape[0]))
                    color_mask = cv2.addWeighted(color_mask, 1, rgb_mask, 1, 0)

                if args.save_binary_mask:
                    mask = masks[i]
                    binary_mask[:, :, common.CLASS_NAMES.index(pred_cls[i])] += mask

            if args.save_color_mask:
                color_mask = cv2.resize(color_mask, (img.shape[1], img.shape[0]))
                p = os.path.join(args.output_dir, 'color_mask')
                if not os.path.isdir(p):
                    os.makedirs(p)
                cv2.imwrite(os.path.join(p, img_paths[idx].split(os.sep)[-1]), color_mask)

            if args.save_binary_mask:    
                binary_mask = binary_mask[:, :, 1:] # remove background rayer
                binary_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[0])) * 255
                np.clip(binary_mask, 0, 255)
                p = os.path.join(args.output_dir, 'binary_mask')
                if not os.path.isdir(p):
                    os.makedirs(p)
                np.savez_compressed(os.path.join(p, img_paths[idx].split(os.sep)[-1])[:-4]+".npz", binary_mask)
