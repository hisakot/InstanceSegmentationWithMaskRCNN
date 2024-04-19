import glob
import json
import os

COCO_JSON = "../breast_surgery2/ss_1st_step_train/COCO_train_annos_re.json"
org = json.load(open(COCO_JSON, encoding="utf-8"))

annotations = org["annotations"]

# rewrite category_id
rewrite = []
e_scalpel = []
scalpel = []
syringe = []
nh = []
pen = []
tweezers = []
forceps = []
hook = []
for i, ann in enumerate(annotations):
    cat = ann["category_id"]
    if cat == 1:
        forceps.append(ann["image_id"])
    elif cat == 2:
        tweezers.append(ann["image_id"])
    elif cat == 3:
        e_scalpel.append(ann["image_id"])
    elif cat == 4:
        scalpel.append(ann["image_id"])
    elif cat == 5:
        hook.append(ann["image_id"])
    elif cat == 6:
        syringe.append(ann["image_id"])
    elif cat == 7:
        nh.append(ann["image_id"])
    elif cat == 8:
        pen.append(ann["image_id"])
    if cat == 1 or cat == 2 or cat == 5:
        ann["category_id"] = 0
    rewrite.append(ann)
print("----------- instance number ----------")
print(len(forceps))
print(len(tweezers))
print(len(e_scalpel))
print(len(scalpel))
print(len(hook))
print(len(syringe))
print(len(nh))
print(len(pen))

print("----------- image number ----------")
print(len(set(forceps)))
print(len(set(tweezers)))
print(len(set(e_scalpel)))
print(len(set(scalpel)))
print(len(set(hook)))
print(len(set(syringe)))
print(len(set(nh)))
print(len(set(pen)))
total_imgs = []
total_imgs.extend(e_scalpel)
total_imgs.extend(scalpel)
total_imgs.extend(syringe)
total_imgs.extend(nh)
total_imgs.extend(pen)
total_imgs.extend(forceps)
total_imgs.extend(tweezers)
total_imgs.extend(hook)
print(len(set(total_imgs)))

org["annotations"] = rewrite

# with open("../breast_surgery2/ss_1st_step_train/COCO_train_annos_5class.json", "w", encoding="utf-8")as outfile:
#     json.dump(org, outfile, sort_keys=True, indent=4, ensure_ascii=False)

