# Marginalia Detection Frankenstein Model COMP4951
The codebase for my semi-supervised marginalia detection segmentation model.
It adapts Cheng et. al.'s Project Marginalia to work as a semi-supervised segmentation model using Shin et. al.'s SemiOVS training pipeline.

# Setup

## Datasets

Download everything from [here](https://drive.google.com/drive/folders/1jDapXx73_s1pxtDCIw1SQPW9GXTq6-LC?usp=drive_link).

For reproducing the extension trial, create a folder "comb" with the sub-folders "images" and "labels". Copy all images from the respective "images" and "labels" folders in "indist" and "ood" into these.

Ensure you have the folder structure for all the data:
```
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
```
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

# Running Scripts

For all the training, use the scripts "BigL.sh", "SmallL.sh", "luu.sh" and "ext_study.sh". After training a model fully, run "cleanup.sh".

For testing, run "test.py", and direct it to "./data/test_images" and give it the "rescaled_data.csv" CSV file.

# References

Cheng, L., Frankemölle, J., Axelsson, A., & Vats, E. (2024, March). Uncovering the Handwritten Text in the Margins: End-to-end Handwritten Text Detection and Recognition. Proceedings of the 8th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2024), 111–120. doi:10.18653/v1/2024.latechclfl-1.12

Shin, W., Kang, J., Jeong, H., Kim, J. S., & Han, S. W. (2025). Leveraging out-of-distribution unlabeled images: Semi-supervised semantic segmentation with an open-vocabulary model. Knowledge-Based Systems, 329, 114289. doi:10.1016/j.knosys.2025.114289