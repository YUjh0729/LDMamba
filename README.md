# LDMamba


## Installation 

Requirements: `Ubuntu 20.04`, `CUDA 11.8`

1. Create a virtual environment: `conda create -n ldmamba python=3.10 -y` and `conda activate ldmamba `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d=1.2.0.post2`
   and `pip install mamba-ssm=1.2.0.post1`
4. Download code: `git clone https://github.com/YUjh0729/LDMamba`
5. `cd LDMamba/ldmamba` and run `pip install -e .`


sanity test: Enter python command-line interface and run.

```bash
import torch
import mamba_ssm
```


## Model Training
Download dataset [here](https://drive.google.com/drive/folders/1DmyIye4Gc9wwaA7MVKFVi-bWD2qQb-qN?usp=sharing) and put them into the `data` folder. LDMamaba is built on the popular [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework. If you want to train LDMamba on your own dataset, please follow this [guideline](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) to prepare the dataset. 

### Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Train 2D models

- Train 2D  model

```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerLDMambaNet
```

### Train 3D models

- Train 3D  model

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerLDMambaNet
```


## Inference

- Predict testing cases with `LDMamba` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainerLDMambaNet --disable_tta
```


> `CONFIGURATION` can be `2d` and `3d_fullres` for 2D and 3D models, respectively.

## Remarks
Path settings

The default data directory for LDMamba is preset to LDMamba/data. Users with existing nnUNet setups who wish to use alternative directories for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` can easily adjust these paths in ldmamba/nnunetv2/path.py to update your specific nnUNet data directory locations, as demonstrated below:

```python
# An example to set other data path,
base = '/home/user_name/Documents/LDMamba/data'
nnUNet_raw = join(base, 'nnUNet_raw') # or change to os.environ.get('nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') # or change to os.environ.get('nnUNet_preprocessed')
nnUNet_results = join(base, 'nnUNet_results') # or change to os.environ.get('nnUNet_results')
```


## Acknowledgements

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [U-Mamba](https://github.com/bowang-lab/U-Mamba) for making their valuable code publicly available.

