import json

# NOTE category = 7
# base_coco = "../data/train/COCO_train_annos.json"
base_coco = "../data/ss_2nd_step_train/COCO_train_annos_re.json"

def modify(base_coco, add_coco):
    base = json.load(open(base_coco, encoding="utf-8"))
    add = json.load(open(add_coco, encoding="utf-8"))

    images = base["images"]
    annotations = base["annotations"]
    num_base_ann = len(annotations)
    num_base_img = len(images)

    add_ann = add["annotations"]
    add_img = add["images"]

    inc = 0
    for ann in add_ann:
        print(inc)
        if len(ann["segmentation"][0]) > 4:
            ann["id"] = num_base_ann + inc
            ann["image_id"] += num_base_img
            base["annotations"].append(ann)
            inc = inc + 1

    for i, img in enumerate(add_img):
        img["id"] = num_base_img + i
        base["images"].append(img)

    with open("../data/ss_2nd_step_train/COCO_train_annos_re.json", "w", encoding="utf-8") as outfile:
        json.dump(base, outfile, sort_keys=True, indent=4, ensure_ascii=False)

add_coco = "../data/add_images/train/COCO_train_annos.json"
modify(base_coco, add_coco)

add_coco = "../data/add_images/val/COCO_val_annos.json"
modify(base_coco, add_coco)
