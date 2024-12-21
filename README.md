# DiffKillR: Killing and Recreating Diffeomorphisms for Cell Annotation in Dense Microscopy Images

### Krishnaswamy Lab, Yale University
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![Github Stars](https://img.shields.io/github/stars/KrishnaswamyLab/DiffKillR.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/DiffKillR/)
[![ArXiv](https://img.shields.io/badge/ArXiv-DiffKillR-firebrick)](https://arxiv.org/abs/2410.03058)

This is the authors' PyTorch implementation of [DiffKillR](https://arxiv.org/abs/2410.03058), ICASSP 2025.

The official version is maintained in the [Lab GitHub repo](https://github.com/KrishnaswamyLab/DiffKillR).


## [New] Usage
Preprocess datasets.
```
cd src/preprocessing
python preprocess_MoNuSeg.py
python preprocess_A28.py
python preprocess_A28_axis.py
```

### Orientation Prediction

Train and test DiffeoInvariantNet. (Remove `--use-wandb` if you don't want to use Weights and Biases.)
```
cd src/
python main_DiffeoInvariantNet.py --dataset-name A28 --dataset-path '$ROOT/data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_patch_96x96/' --DiffeoInvariantNet-model AutoEncoder --use-wandb --wandb-username yale-cl2482
```

Train and test DiffeoMappingNet.
```
cd src/
python main_DiffeoMappingNet.py --dataset-name A28 --dataset-path '$ROOT/data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_patch_96x96/' --DiffeoMappingNet-model VoxelMorph --use-wandb --wandb-username yale-cl2482
```

### Segmentation on MoNuSeg

Train and test DiffeoInvariantNet.
```
cd src/
python main_DiffeoInvariantNet.py --dataset-name MoNuSeg --dataset-path '$ROOT/data/MoNuSeg/MoNuSegByCancer_patch_96x96/' --organ Breast --percentage 10 --random-seed 1
python main_DiffeoInvariantNet.py --dataset-name MoNuSeg --dataset-path '$ROOT/data/MoNuSeg/MoNuSegByCancer_patch_96x96/' --organ Colon --percentage 10 --random-seed 1
python main_DiffeoInvariantNet.py --dataset-name MoNuSeg --dataset-path '$ROOT/data/MoNuSeg/MoNuSegByCancer_patch_96x96/' --organ Prostate --percentage 10 --random-seed 1
```

Train and test DiffeoMappingNet.
```
cd src/
python main_DiffeoMappingNet.py --dataset-name MoNuSeg --dataset-path '$ROOT/data/MoNuSeg/MoNuSegByCancer_patch_96x96/' --organ Breast --percentage 10 --random-seed 1
python main_DiffeoMappingNet.py --dataset-name MoNuSeg --dataset-path '$ROOT/data/MoNuSeg/MoNuSegByCancer_patch_96x96/' --organ Colon --percentage 10 --random-seed 1
python main_DiffeoMappingNet.py --dataset-name MoNuSeg --dataset-path '$ROOT/data/MoNuSeg/MoNuSegByCancer_patch_96x96/' --organ Prostate --percentage 10 --random-seed 1
```

Run Inference
```
cd src
python main_inference_segmentation.py --organ Breast --use-gt-loc --random-seed 1
python main_inference_segmentation.py --organ Colon --use-gt-loc --random-seed 1
python main_inference_segmentation.py --organ Prostate --use-gt-loc --random-seed 1
python main_inference_segmentation.py --organ Breast --random-seed 1
python main_inference_segmentation.py --organ Colon --random-seed 1
python main_inference_segmentation.py --organ Prostate --random-seed 1
```

### Comparison
1. First train/infer the models. Can refer to `bash/baseline_medt.sh`, `bash/baseline_psm.sh`, `bash/baseline_sam.sh`.

2. Then, stitch the images and run evaluation.
```
cd comparison/eval/
python stitch_patches.py
python evaluate_monuseg.py
python evaluate_glysac.py
```



## Preparation

#### To use SAM.
```
## under `comparison/SAM/checkpoints/`
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### To Use SAM2.
```
## under `comparison/SAM2/checkpoints/`
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

#### To use MedSAM.
```
## under `comparison/MedSAM/checkpoints/`
download from https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view
```

#### To use SAM-Med2D.
```
## under `comparison/SAM_Med2D/checkpoints/`
download from https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view
```



## Naming Conventions
```
    # The output of the model will be saved in the following directory
    # e.g. 0.100_Colon_m2_MoNuSeg_depth5_seed1_SimCLR
    # -- reg2seg: Reg2Seg results
    # -- reg2seg.ckpt: Reg2Seg model
    # -- aiae.ckpt: AIAE model
    # -- test_pairs.csv: test pairs
    # -- train_pairs.csv: training pairs
    model_name = f'{config.percentage:.3f}_{config.organ}_m{config.multiplier}_MoNuSeg_depth{config.depth}_seed{config.random_seed}_{config.latent_loss}'
```

## Train on MoNuSeg
```
# prepare data
cd src/
# This will extract annotations & patches from the MoNuSeg dataset
python preprocessing/prepare_MoNuseg.py --patch_size 96 --aug_patch_size 32
# This will subsample and augment the dataset, and output data yaml config file in '../config/MoNuSeg_data.yaml'
python preprocessing/augment_MoNuseg.py \
                  --patch_size {patch_size} \
                  --augmented_patch_size {aug_patch_size} \
                  --percentage {percent} \
                  --multiplier {multiplier} \
                  --organ {organ_type}

# train AIAE
python train_unsupervised_AE.py \
              --mode train \
              --data-config  '../config/MoNuSeg_data.yaml' \
              --model-config '../config/MoNuSeg_AIAE.yaml' \
              --num-workers {num_workers}

# infer & generate training & test pairs for Reg2Seg
python train_unsupervised_AE.py \
              --mode infer \
              --data-config '../config/MoNuSeg_data.yaml' \
              --model-config '../config/MoNuSeg_reg2seg.yaml' \
              --num-workers {num_workers}

# train Reg2Seg using matched pairs
python train_reg2seg.py --mode train --config ../config/MoNuSeg_reg2seg.yaml

# infer segmentation using Reg2Seg
python train_reg2seg.py --config ../config/MoNuSeg_reg2seg.yaml --mode infer

```

### External Dataset

#### TissueNet
```
cd external_data/TissueNet
# Download from https://datasets.deepcell.org/data
unzip tissuenet_v1.1.zip

python preprocess_tissuenet.py
```


#### MoNuSeg
```
cd external_data/MoNuSeg

cd src/preprocessing
python preprocess_MoNuSeg.py
```


#### GLySAC
```
cd src/preprocessing
python preprocess_GLySAC.py
```

### Environment
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
# Optional: Update to libmamba solver.
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --name cellseg pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia -c anaconda -c conda-forge

conda activate cellseg
conda install -c anaconda scikit-image scikit-learn pillow matplotlib seaborn tqdm
# conda install -c conda-forge libstdcxx-ng=12
python -m pip install antspyx
# python -m pip install dipy
python -m pip install opencv-python
python -m pip install python-dotenv

# MoNuSeg
python -m pip install xmltodict

# PSM
python -m pip install tensorboardX
python -m pip install shapely
python -m pip install ml_collections
python -m pip install ttach

## LACSS
#python -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install jax==0.4.24
python -m pip install lacss
#python -m pip install ml_dtypes==0.2.0

## StarDist
python -m pip install stardist
python -m pip install tensorflow

# For SAM
python -m pip install git+https://github.com/facebookresearch/segment-anything.git

# For SAM2
python -m pip install git+https://github.com/facebookresearch/segment-anything-2.git

# For MedSAM
python -m pip install git+https://github.com/bowang-lab/MedSAM.git

# For SAM-Med2D
python -m pip install albumentations
python -m pip install scikit-learn==1.1.3  # need to downgrade to 1.1.3


# Export CuDNN
# echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

```

## Train Augmentation-agonostic network
```
python train_AE.py --mode train --config {config} --num-workers 4
```



```
cd src/scripts/
python match_cells.py --config ../../config/aug_AutoEncoder_depth5_seed1_simCLR.yaml
```
