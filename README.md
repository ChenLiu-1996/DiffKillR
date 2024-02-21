# Semi-supervised Cell Segmentation with Diffeomorphism Generalization

<!-- [![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![Github Stars](https://img.shields.io/github/stars/ChenLiu-1996/DiffusionSpectralEntropy.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/DiffusionSpectralEntropy/) -->



## Preparation

## Train on MoNuSeg
```
# prepare data
cd src/
# This will extract annotations & patches from the MoNuSeg dataset
python preprocessing/prepare_MoNuseg.py --patch_size 96 --aug_patch_size 32
# This will subsample and augment the dataset, and output data yaml config
python preprocessing/augment_MoNuseg.py \
                  --patch_size {patch_size} \
                  --augmented_patch_size {aug_patch_size} \
                  --percentage {percent} \
                  --multiplier {multiplier} \
                  --organ {organ_type}

# train AIAE & generate training pairs for Reg2Seg
python train_unsupervised_AE.py \
              --mode train \
              --data-config {data_config} \
              --model-config {model_config} \
              --num-workers {num_workers}

# infer test matched pairs using AIAE
python train_unsupervised_AE.py \
              --mode infer \
              --data-config {data_config} \
              --model-config {model_config} \
              --num-workers {num_workers}

# train Reg2Seg using matched pairs
python train_reg2seg.py --config ../config/MoNuSeg_reg2seg.yaml --mode train

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

# MoNuSeg
python -m pip install xmltodict

# PSM
python -m pip install tensorboardX
python -m pip install shapely
python -m pip install ml_collections
python -m pip install ttach

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