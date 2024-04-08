#!/bin/bash

#SBATCH --job-name=medt-mn
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=2-00:00:00
#SBATCH --mem=30G
#SBATCH --mail-type=ALL

### For Farnam
# module purge
# module load miniconda
# module load CUDAcore/11.2.2 cuDNN/8.1.1.33-CUDA-11.2.2
# module load GCC/7.3.0
# module load git/2.30.0-GCCcore-10.2.0-nodocs

### For McCleary
module purge
module load miniconda
module load CUDAcore/11.1.1 cuDNN/8.0.5.39-CUDA-11.1.1
module load GCC/10.2.0
module load git/2.28.0-GCCcore-10.2.0-nodocs

source ~/.zshrc
conda activate cellseg

cd /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/CellSeg/comparison/MedT/

for i in $(seq 1 3);
do
    for cancer in breast colon prostate;
    do
        time python train.py --train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --direc "../results/MoNuSegByCancer_200x200/$cancer/MedT_seed$i/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "MedT" \
        --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

        time python test.py --loaddirec "../results/MoNuSegByCancer_200x200/$cancer/MedT_seed$i/final_model.pth" \
        --train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/test/" \
        --direc "../results/MoNuSegByCancer_200x200/$cancer/MedT_seed$i/" \
        --batch_size 1 --modelname "MedT" --imgsize 200 --gray "no"

        time python train_unet.py --train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --direc "../results/MoNuSegByCancer_200x200/$cancer/UNet_seed$i/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "UNet" \
        --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

        time python test_unet.py --loaddirec "../results/MoNuSegByCancer_200x200/$cancer/UNet_seed$i/final_model.pth" \
        --train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/test/" \
        --direc "../results/MoNuSegByCancer_200x200/$cancer/UNet_seed$i/" \
        --batch_size 1 --modelname "UNet" --imgsize 200 --gray "no"

        time python train_nnunet.py --train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --direc "../results/MoNuSegByCancer_200x200/$cancer/nnUNet_seed$i/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "nnUNet" \
        --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

        time python test_nnunet.py --loaddirec "../results/MoNuSegByCancer_200x200/$cancer/nnUNet_seed$i/final_model.pth" \
        --train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/train/" \
        --val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/$cancer/test/" \
        --direc "../results/MoNuSegByCancer_200x200/$cancer/nnUNet_seed$i/" \
        --batch_size 1 --modelname "nnUNet" --imgsize 200 --gray "no"
    done
done


# for i in $(seq 1 3);
# do
#     for cancer in normal tumor;
#     do
#         time python train.py --train_dataset "../../external_data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
#         --val_dataset "../../external_data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
#         --direc "../results/GLySACByTumor_200x200/$cancer/MedT_seed$i/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "MedT" \
#         --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

#         time python test.py --loaddirec "../results/GLySACByTumor_200x200/$cancer/MedT_seed$i/final_model.pth" \
#         --train_dataset "../../external_data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
#         --val_dataset "../../external_data/GLySAC/GLySACByTumor_200x200/$cancer/test/" \
#         --direc "../results/GLySACByTumor_200x200/$cancer/MedT_seed$i/" \
#         --batch_size 1 --modelname "MedT" --imgsize 200 --gray "no"

#         time python train_unet.py --train_dataset "../../external_data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
#         --val_dataset "../../external_data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
#         --direc "../results/GLySACByTumor_200x200/$cancer/UNet_seed$i/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "UNet" \
#         --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

#         time python test_unet.py --loaddirec "../results/GLySACByTumor_200x200/$cancer/UNet_seed$i/final_model.pth" \
#         --train_dataset "../../external_data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
#         --val_dataset "../../external_data/GLySAC/GLySACByTumor_200x200/$cancer/test/" \
#         --direc "../results/GLySACByTumor_200x200/$cancer/UNet_seed$i/" \
#         --batch_size 1 --modelname "UNet" --imgsize 200 --gray "no"

#         time python train_nnunet.py --train_dataset "../../external_data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
#         --val_dataset "../../external_data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
#         --direc "../results/GLySACByTumor_200x200/$cancer/nnUNet_seed$i/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "nnUNet" \
#         --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

#         time python test_nnunet.py --loaddirec "../results/GLySACByTumor_200x200/$cancer/nnUNet_seed$i/final_model.pth" \
#         --train_dataset "../../external_data/GLySAC/GLySACByTumor_200x200/$cancer/train/" \
#         --val_dataset "../../external_data/GLySAC/GLySACByTumor_200x200/$cancer/test/" \
#         --direc "../results/GLySACByTumor_200x200/$cancer/nnUNet_seed$i/" \
#         --batch_size 1 --modelname "nnUNet" --imgsize 200 --gray "no"
#     done
# done
