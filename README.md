# Marginalia Detection Frankenstein Model COMP4951
The codebase for my semi-supervised marginalia detection segmentation model.
It adapts Cheng et. al.'s Project Marginalia to work as a semi-supervised segmentation model using Shin et. al.'s SemiOVS training pipeline.

# Setup

## Datasets

Download everything from [here](https://drive.google.com/drive/folders/1jDapXx73_s1pxtDCIw1SQPW9GXTq6-LC?usp=drive_link).

For reproducing the extension trial, create a folder "comb" with the sub-folders "images" and "labels". Copy all images from the respective "images" and "labels" folders in "indist" and "ood" into these.

Ensure you have the folder structure for all the data:
data -
    indist -
        images -
        labels -
    ood -
        images -
        labels -
    comb -
        images -
        labels -
    test_images -

## Conda Environment
```
conda create -n frankenstein python=3.9 -y
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python -m pip install -e .
```
You may need to manually install the cuda development tools in your conda environment. Cuda version 11.6 needed.

## Making Scripts Executable
```
chmod u+x ./script/*.sh
```
Please execute scripts from the root of the repository, so that paths are correctly read.