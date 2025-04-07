<h1 align="center">
[ICASSP 2025] DiffKillR
</h1>

<p align="center">
<strong>Killing and Recreating Diffeomorphisms for Cell Annotation in Dense Microscopy Images</strong>
</p>

<div align="center">

[![ArXiv](https://img.shields.io/badge/ArXiv-DiffKillR-firebrick)](https://arxiv.org/abs/2410.03058)
[![Slides](https://img.shields.io/badge/Slides-yellow)](https://chenliu-1996.github.io/slides/DiffKillR_slides.pdf)
[![ICASSP](https://img.shields.io/badge/ICASSP-blue)](https://ieeexplore.ieee.org/abstract/document/10888526)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social)](https://twitter.com/KrishnaswamyLab)
[![Twitter](https://img.shields.io/twitter/follow/ChenLiu-1996.svg?style=social)](https://twitter.com/ChenLiu_1996)
[![Twitter](https://img.shields.io/twitter/follow/DanqiLiao.svg?style=social)](https://x.com/DanqiLiao73090)
[![Github Stars](https://img.shields.io/github/stars/KrishnaswamyLab/DiffKillR.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/DiffKillR/)

</div>

### Krishnaswamy Lab, Yale University

This is the authors' PyTorch implementation of [DiffKillR](https://arxiv.org/abs/2410.03058), ICASSP 2025.

The official version is maintained in the [Lab GitHub repo](https://github.com/KrishnaswamyLab/DiffKillR).


## Citation
```
@inproceedings{liu2025diffkillr,
  title={Diffkillr: Killing and recreating diffeomorphisms for cell annotation in dense microscopy images},
  author={Liu, Chen and Liao, Danqi and Parada-Mayorga, Alejandro and Ribeiro, Alejandro and DiStasio, Marcello and Krishnaswamy, Smita},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025},
  organization={IEEE}
}

@article{liu2024diffkillr,
  title={Diffkillr: Killing and recreating diffeomorphisms for cell annotation in dense microscopy images},
  author={Liu, Chen and Liao, Danqi and Parada-Mayorga, Alejandro and Ribeiro, Alejandro and DiStasio, Marcello and Krishnaswamy, Smita},
  journal={arXiv preprint arXiv:2410.03058},
  year={2024}
}
```

## Usage
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
```

Train and test DiffeoMappingNet.
```
cd src/
python main_DiffeoMappingNet.py --dataset-name MoNuSeg --dataset-path '$ROOT/data/MoNuSeg/MoNuSegByCancer_patch_96x96/' --organ Breast --percentage 10 --random-seed 1
```

Run Inference
```
python main_inference_segmentation.py --organ Breast --use-gt-loc --random-seed 1
python main_inference_segmentation.py --organ Breast --random-seed 1
```

### Comparison
1. First train/infer the models.

<details>
<summary>1.1 MedT, UNet, nnUNet.</summary>

```
cd /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/DiffKillR/comparison/MedT/

for i in $(seq 1 3);
do
    for cancer in Bladdar Brain Breast Colon Kidney Liver Lung Prostate Stomach;
    do
        time python train.py --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --direc "../results/MoNuSegByCancer_200x200/$cancer/MedT_seed$i/" --batch_size 4 --epoch 100 --save_freq 100 --modelname "MedT" \
        --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

        time python test.py --loaddirec "../results/MoNuSegByCancer_200x200/$cancer/MedT_seed$i/final_model.pth" \
        --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/test/" \
        --direc "../results/MoNuSegByCancer_200x200/$cancer/MedT_seed$i/" \
        --batch_size 4 --modelname "MedT" --imgsize 200 --gray "no"

        time python train_unet.py --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --direc "../results/MoNuSegByCancer_200x200/$cancer/UNet_seed$i/" --batch_size 4 --epoch 100 --save_freq 100 --modelname "UNet" \
        --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

        time python test_unet.py --loaddirec "../results/MoNuSegByCancer_200x200/$cancer/UNet_seed$i/final_model.pth" \
        --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/test/" \
        --direc "../results/MoNuSegByCancer_200x200/$cancer/UNet_seed$i/" \
        --batch_size 4 --modelname "UNet" --imgsize 200 --gray "no"

        time python train_nnunet.py --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --direc "../results/MoNuSegByCancer_200x200/$cancer/nnUNet_seed$i/" --batch_size 4 --epoch 100 --save_freq 100 --modelname "nnUNet" \
        --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

        time python test_nnunet.py --loaddirec "../results/MoNuSegByCancer_200x200/$cancer/nnUNet_seed$i/final_model.pth" \
        --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/test/" \
        --direc "../results/MoNuSegByCancer_200x200/$cancer/nnUNet_seed$i/" \
        --batch_size 4 --modelname "nnUNet" --imgsize 200 --gray "no"
    done
done

for i in $(seq 1 3);
do
    for cancer in Tumor Normal;
    do
        time python train.py --train_dataset "../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
        --val_dataset "../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
        --direc "../results/GLySACByTumor_200x200/$cancer/MedT_seed$i/" --batch_size 4 --epoch 100 --save_freq 100 --modelname "MedT" \
        --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

        time python test.py --loaddirec "../results/GLySACByTumor_200x200/$cancer/MedT_seed$i/final_model.pth" \
        --train_dataset "../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
        --val_dataset "../../data/GLySAC/GLySACByTumor_200x200/$cancer/test/" \
        --direc "../results/GLySACByTumor_200x200/$cancer/MedT_seed$i/" \
        --batch_size 4 --modelname "MedT" --imgsize 200 --gray "no"

        time python train_unet.py --train_dataset "../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
        --val_dataset "../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
        --direc "../results/GLySACByTumor_200x200/$cancer/UNet_seed$i/" --batch_size 4 --epoch 100 --save_freq 100 --modelname "UNet" \
        --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

        time python test_unet.py --loaddirec "../results/GLySACByTumor_200x200/$cancer/UNet_seed$i/final_model.pth" \
        --train_dataset "../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
        --val_dataset "../../data/GLySAC/GLySACByTumor_200x200/$cancer/test/" \
        --direc "../results/GLySACByTumor_200x200/$cancer/UNet_seed$i/" \
        --batch_size 4 --modelname "UNet" --imgsize 200 --gray "no"

        time python train_nnunet.py --train_dataset "../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
        --val_dataset "../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
        --direc "../results/GLySACByTumor_200x200/$cancer/nnUNet_seed$i/" --batch_size 4 --epoch 100 --save_freq 100 --modelname "nnUNet" \
        --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

        time python test_nnunet.py --loaddirec "../results/GLySACByTumor_200x200/$cancer/nnUNet_seed$i/final_model.pth" \
        --train_dataset "../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
        --val_dataset "../../data/GLySAC/GLySACByTumor_200x200/$cancer/test/" \
        --direc "../results/GLySACByTumor_200x200/$cancer/nnUNet_seed$i/" \
        --batch_size 4 --modelname "nnUNet" --imgsize 200 --gray "no"
    done
done
```
</details>

<details>
<summary>1.2 PSM.</summary>

```
for i in $(seq 1 3);
do
    for cancer in Bladdar Brain Breast Colon Kidney Liver Lung Prostate Stomach;
    do
        time python main_train_test.py --seed $i --mode 'train_base' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/$cancer --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/test/
        time python main_train_test.py --seed $i --mode 'generate_label' --method 'gradcam' --dataset_name MoNuSegByCancer_200x200/$cancer --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/test/
        time python main_train_test.py --seed $i --mode 'train_second_stage' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/$cancer --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/test/
        time python main_train_test.py --seed $i --mode 'generate_voronoi' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/$cancer --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/test/
        time python main_train_test.py --seed $i --mode 'train_final_stage' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/$cancer --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/test/
        time python main_train_test.py --seed $i --mode 'test' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/$cancer --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/test/
    done
done

for i in $(seq 1 3);
do
    for cancer in Tumor Normal;
    do
        time python main_train_test.py --seed $i --mode 'train_base' --crop_edge_size 200 --dataset_name GLySACByTumor_200x200/$cancer --data_train ../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/ --data_test ../../data/GLySAC/GLySACByTumor_200x200/$cancer/test/
        time python main_train_test.py --seed $i --mode 'generate_label' --method 'gradcam' --dataset_name GLySACByTumor_200x200/$cancer --data_train ../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/ --data_test ../../data/GLySAC/GLySACByTumor_200x200/$cancer/test/
        time python main_train_test.py --seed $i --mode 'train_second_stage' --crop_edge_size 200 --dataset_name GLySACByTumor_200x200/$cancer --data_train ../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/ --data_test ../../data/GLySAC/GLySACByTumor_200x200/$cancer/test/
        time python main_train_test.py --seed $i --mode 'generate_voronoi' --crop_edge_size 200 --dataset_name GLySACByTumor_200x200/$cancer --data_train ../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/ --data_test ../../data/GLySAC/GLySACByTumor_200x200/$cancer/test/
        time python main_train_test.py --seed $i --mode 'train_final_stage' --crop_edge_size 200 --dataset_name GLySACByTumor_200x200/$cancer --data_train ../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/ --data_test ../../data/GLySAC/GLySACByTumor_200x200/$cancer/test/
        time python main_train_test.py --seed $i --mode 'test' --crop_edge_size 200 --dataset_name GLySACByTumor_200x200/$cancer --data_train ../../data/GLySAC/GLySACByTumor_200x200/$cancer/train/ --data_test ../../data/GLySAC/GLySACByTumor_200x200/$cancer/test/
    done
done
```
</details>

<details>
<summary>1.3 SAM, MedSAM, SAM-Med2D, SAM2.</summary>

```
for cancer in Bladdar Brain Breast Colon Kidney Liver Lung Prostate Stomach;
do
    cd /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/DiffKillR/comparison/SAM/
    time python test.py --val_dataset "../../data/MoNuSeg/MoNuSegByCancer/$cancer/test/" \
    --direc "../results/MoNuSegByCancer/$cancer/SAM/" --imgsize 1000 --gray "no"

    cd /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/DiffKillR/comparison/SAM2/
    time python test.py --val_dataset "../../data/MoNuSeg/MoNuSegByCancer/$cancer/test/" \
    --direc "../results/MoNuSegByCancer/$cancer/SAM2/" --imgsize 1000 --gray "no"

    cd /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/DiffKillR/comparison/SAM_Med2D/
    time python test.py --val_dataset "../../data/MoNuSeg/MoNuSegByCancer/$cancer/test/" \
    --direc "../results/MoNuSegByCancer/$cancer/SAM_Med2D/" --imgsize 1000 --gray "no"

    cd /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/DiffKillR/comparison/MedSAM/
    time python test.py --val_dataset "../../data/MoNuSeg/MoNuSegByCancer/$cancer/test/" \
    --direc "../results/MoNuSegByCancer/$cancer/MedSAM/" --imgsize 1000 --gray "no"
done

for cancer in Tumor Normal;
do
    cd /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/DiffKillR/comparison/SAM/
    time python test.py --val_dataset "../../data/GLySAC/GLySACByTumor/$cancer/test/" \
    --direc "../results/GLySACByTumor/$cancer/SAM/" --imgsize 1000 --gray "no"

    cd /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/DiffKillR/comparison/SAM2/
    time python test.py --val_dataset "../../data/GLySAC/GLySACByTumor/$cancer/test/" \
    --direc "../results/GLySACByTumor/$cancer/SAM2/" --imgsize 1000 --gray "no"

    cd /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/DiffKillR/comparison/SAM_Med2D/
    time python test.py --val_dataset "../../data/GLySAC/GLySACByTumor/$cancer/test/" \
    --direc "../results/GLySACByTumor/$cancer/SAM_Med2D/" --imgsize 1000 --gray "no"

    cd /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/DiffKillR/comparison/MedSAM/
    time python test.py --val_dataset "../../data/GLySAC/GLySACByTumor/$cancer/test/" \
    --direc "../results/GLySACByTumor/$cancer/MedSAM/" --imgsize 1000 --gray "no"
done
```
</details>


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
python -m pip install simple-lama-inpainting

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

