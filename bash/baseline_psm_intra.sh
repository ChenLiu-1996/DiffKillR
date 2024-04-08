#!/bin/bash

#SBATCH --job-name=intra-psm
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

cd /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/CellSeg/comparison/PSM/

for i in $(seq 1 3);
do
    for cancer in breast colon prostate;
    do
        for pct in MoNuSegByCancer_intraimage5pct_200x200 MoNuSegByCancer_intraimage20pct_200x200 MoNuSegByCancer_intraimage50pct_200x200;
        do
            # Iterate over possible files. Please change the range to the max number of files.
            for img_cnt in $(seq 0 10);
            do
                if [ -d "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/" ]; then
                    time python main_train_test.py --seed $i --mode 'train_base' --crop_edge_size 200 --dataset_name $pct/${cancer}_img${img_cnt} --data_train ../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/ --data_test ../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_test/
                    time python main_train_test.py --seed $i --mode 'generate_label' --method 'gradcam' --dataset_name $pct/${cancer}_img${img_cnt} --data_train ../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/ --data_test ../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_test/
                    time python main_train_test.py --seed $i --mode 'train_second_stage' --crop_edge_size 200 --dataset_name $pct/${cancer}_img${img_cnt} --data_train ../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/ --data_test ../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_test/
                    time python main_train_test.py --seed $i --mode 'generate_voronoi' --crop_edge_size 200 --dataset_name $pct/${cancer}_img${img_cnt} --data_train ../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/ --data_test ../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_test/
                    time python main_train_test.py --seed $i --mode 'train_final_stage' --crop_edge_size 200 --dataset_name $pct/${cancer}_img${img_cnt} --data_train ../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/ --data_test ../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_test/
                    time python main_train_test.py --seed $i --mode 'test' --crop_edge_size 200 --dataset_name $pct/${cancer}_img${img_cnt} --data_train ../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/ --data_test ../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_test/
                fi
            done
        done
    done
done

for i in $(seq 1 3);
do
    for cancer in normal tumor;
    do
        for pct in GLySACByTumor_intraimage5pct_200x200 GLySACByTumor_intraimage20pct_200x200 GLySACByTumor_intraimage50pct_200x200;
        do
            # Iterate over possible files. Please change the range to the max number of files.
            for img_cnt in $(seq 0 10);
            do
                if [ -d "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/" ]; then
                    time python main_train_test.py --seed $i --mode 'train_base' --crop_edge_size 200 --dataset_name $pct/${cancer}_img${img_cnt} --data_train ../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/ --data_test ../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_test/
                    time python main_train_test.py --seed $i --mode 'generate_label' --method 'gradcam' --dataset_name $pct/${cancer}_img${img_cnt} --data_train ../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/ --data_test ../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_test/
                    time python main_train_test.py --seed $i --mode 'train_second_stage' --crop_edge_size 200 --dataset_name $pct/${cancer}_img${img_cnt} --data_train ../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/ --data_test ../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_test/
                    time python main_train_test.py --seed $i --mode 'generate_voronoi' --crop_edge_size 200 --dataset_name $pct/${cancer}_img${img_cnt} --data_train ../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/ --data_test ../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_test/
                    time python main_train_test.py --seed $i --mode 'train_final_stage' --crop_edge_size 200 --dataset_name $pct/${cancer}_img${img_cnt} --data_train ../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/ --data_test ../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_test/
                    time python main_train_test.py --seed $i --mode 'test' --crop_edge_size 200 --dataset_name $pct/${cancer}_img${img_cnt} --data_train ../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/ --data_test ../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_test/
                fi
            done
        done
    done
done
