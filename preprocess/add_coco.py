import json

base_coco = "../data/ss_2nd_step_train/COCO_train_annos_re.json"
new_coco = "../data/ss_2nd_step_train/COCO_train_annos.json"
add_coco = "../data/add_images/train/COCO_train_annos.json" # both
# add_coco = "../data/add_images/val/COCO_val_annos.json" # both
# category = 7

base = json.load(open(base_coco, encoding="utf-8"))
add = json.load(open(add_coco, encoding="utf-8"))


# key : images
images = base["images"]
add_img = add["images"]
for i in range(len(add_img)):
    add_img[i]["id"] = len(images) + i
    base["images"] .append(add_img[i])

# key : annotations
annotations = base["annotations"]
add_anno = add["annotations"]
for a in range(len(add_anno)):
    add_anno[a]["id"] = len(annotations) + a
    add_anno[a]["image_id"] += len(images)
    base["annotations"].append(add_anno[a])

with open(new_coco, "w", encoding="utf-8") as f:
    json.dump(base, f, sort_keys=True, indent=4, ensure_ascii=False)

# remove segmentation length <= 4
annotations = base["annotations"]
images = base["images"]

image_ids = []
tmp_anns = []
for ann in annotations:
    if len(ann["segmentation"][0]) <= 4:
        image_ids.append(ann["image_id"])
    else:
        tmp_anns.append(ann)

remove = False
new_ann = []
for ann in tmp_anns:
    for image_id in image_ids:
        if ann["image_id"] == image_id:
             remove = True
	if remove == False:
            new_ann.append(ann)
        remove = False

remove = False
new_img = []
for img in images:
    for image_id in image_ids:
        if img["id"] == image_id:
            remove = True
        if remove == False:
            new_img.append(img)
        remove = False

print("images: ", len(images), "new_images: ", len(new_img))
print("annotations: ", len(annotations), "new_annotations: ", len(new_ann))

base["annotations"] = new_ann
base["images"] = new_img
with open(base_coco, "w", encoding="utf-8") as outfile:
    jdon.dump(base, outfile, sort_keys=True, indent=4, ensure_ascii=False)
