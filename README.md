# Instance Segmentation with MaskR-CNN and ViT
Using this to detect surgical tools or inner body tissues.

## Installation
### Requirements
- Windows10
- Python 3.7
- CUDA 12.1

### Prepare
- Clone the repository locally:
```
git clone https://github.com/hisakot/InstanceSegmentationWithMaskRCNN.git
```

- Create a conda virtual environment and active it:
```
conda create --name InstanceSegmentation python=3.7.3
conda activate InstanceSegmentation
```

- Install Python requirements:
To install pytorch, torchvision and torchaudio with using cuda,
   click [INSTALL PYTORCH](https://pytorch.org) and choose a command suit for your environment.
```
pip install -r requirements.txt
```

## Demo
- common.py

Firstly, rewrite TEST_IMG_PATH.
optionally, rewrite SAVE_COLOR_DIR and SAVE_BINARY_DIR.

- demo.py
```
python demo.py --model_name * model name *
python demo.py --model_name *model name* --save_bbox --save_color_mask --save_binary_mask --save_coco_json --output_dir *output dir name*
```

## Training
- main.py
- main_cv.py

You can train `data` folder's dataset.
In common.py, train and test dataset's root path and annotaion file path are be able to changed.

Optionally, you can use wandb.

If you want to use checkpoint model, please add load_model option and model name.

Early stopping epoch is set to 50 by default, you can set your own epoch.

```
python main.py
python main.py --wandb --load_model *model name* --early_stopping 100
```

If you want to use cross validation, main_cv.py is usable.
In common.py, KFOLD_NUM is able to change, dafault is 5.
Options are the same with main.py.

## Preprocess
- via2coco

VIA Image Annotator was used to annotate surgical field.
If you import annotation data as COCO format, this part is not used.

整えていない
