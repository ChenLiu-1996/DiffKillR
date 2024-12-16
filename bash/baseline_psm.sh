#!/bin/bash

#SBATCH --job-name=psm
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

cd /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/DiffKillR/comparison/PSM/

for i in $(seq 1 3);
do
    for cancer in Breast Colon Prostate;
    do
        time python main_train_test.py --seed $i --mode 'train_base' --crop_edge_size 1000 --dataset_name MoNuSegByCancer/$cancer --data_train ../../data/MoNuSeg/MoNuSegByCancer/$cancer/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer/$cancer/test/
        time python main_train_test.py --seed $i --mode 'generate_label' --method 'gradcam' --dataset_name MoNuSegByCancer/$cancer --data_train ../../data/MoNuSeg/MoNuSegByCancer/$cancer/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer/$cancer/test/
        time python main_train_test.py --seed $i --mode 'train_second_stage' --crop_edge_size 1000 --dataset_name MoNuSegByCancer/$cancer --data_train ../../data/MoNuSeg/MoNuSegByCancer/$cancer/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer/$cancer/test/
        time python main_train_test.py --seed $i --mode 'generate_voronoi' --crop_edge_size 1000 --dataset_name MoNuSegByCancer/$cancer --data_train ../../data/MoNuSeg/MoNuSegByCancer/$cancer/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer/$cancer/test/
        time python main_train_test.py --seed $i --mode 'train_final_stage' --crop_edge_size 1000 --dataset_name MoNuSegByCancer/$cancer --data_train ../../data/MoNuSeg/MoNuSegByCancer/$cancer/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer/$cancer/test/
        time python main_train_test.py --seed $i --mode 'test' --crop_edge_size 1000 --dataset_name MoNuSegByCancer/$cancer --data_train ../../data/MoNuSeg/MoNuSegByCancer/$cancer/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer/$cancer/test/
    done
done

# for i in $(seq 1 3);
# do
#     for cancer in normal tumor;
#     do
#         time python main_train_test.py --seed $i --mode 'train_base' --crop_edge_size 1000 --dataset_name GLySACByTumor/$cancer --data_train ../../data/GLySAC/GLySACByTumor/$cancer/train/ --data_test ../../data/GLySAC/GLySACByTumor/$cancer/test/
#         time python main_train_test.py --seed $i --mode 'generate_label' --method 'gradcam' --dataset_name GLySACByTumor/$cancer --data_train ../../data/GLySAC/GLySACByTumor/$cancer/train/ --data_test ../../data/GLySAC/GLySACByTumor/$cancer/test/
#         time python main_train_test.py --seed $i --mode 'train_second_stage' --crop_edge_size 1000 --dataset_name GLySACByTumor/$cancer --data_train ../../data/GLySAC/GLySACByTumor/$cancer/train/ --data_test ../../data/GLySAC/GLySACByTumor/$cancer/test/
#         time python main_train_test.py --seed $i --mode 'generate_voronoi' --crop_edge_size 1000 --dataset_name GLySACByTumor/$cancer --data_train ../../data/GLySAC/GLySACByTumor/$cancer/train/ --data_test ../../data/GLySAC/GLySACByTumor/$cancer/test/
#         time python main_train_test.py --seed $i --mode 'train_final_stage' --crop_edge_size 1000 --dataset_name GLySACByTumor/$cancer --data_train ../../data/GLySAC/GLySACByTumor/$cancer/train/ --data_test ../../data/GLySAC/GLySACByTumor/$cancer/test/
#         time python main_train_test.py --seed $i --mode 'test' --crop_edge_size 1000 --dataset_name GLySACByTumor/$cancer --data_train ../../data/GLySAC/GLySACByTumor/$cancer/train/ --data_test ../../data/GLySAC/GLySACByTumor/$cancer/test/
#     done
# done
