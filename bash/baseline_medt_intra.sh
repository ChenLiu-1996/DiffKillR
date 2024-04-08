#!/bin/bash

#SBATCH --job-name=intra-medt1
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
        for pct in MoNuSegByCancer_intraimage5pct_200x200 MoNuSegByCancer_intraimage20pct_200x200 MoNuSegByCancer_intraimage50pct_200x200;
        do
            # Iterate over possible files. Please change the range to the max number of files.
            for img_cnt in $(seq 0 15);
            do
                if [ -d "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/" ]; then
                    time python train.py --train_dataset "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/" \
                    --val_dataset "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/" \
                    --direc "../results/$pct/${cancer}_img${img_cnt}/MedT_seed$i/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "MedT" \
                    --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

                    time python test.py --loaddirec "../results/$pct/${cancer}_img${img_cnt}/MedT_seed$i/final_model.pth" \
                    --train_dataset "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/" \
                    --val_dataset "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_test/" \
                    --direc "../results/$pct/${cancer}_img${img_cnt}/MedT_seed$i/" \
                    --batch_size 1 --modelname "MedT" --imgsize 200 --gray "no"

                    time python train_unet.py --train_dataset "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/" \
                    --val_dataset "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/" \
                    --direc "../results/$pct/${cancer}_img${img_cnt}/UNet_seed$i/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "UNet" \
                    --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

                    time python test_unet.py --loaddirec "../results/$pct/${cancer}_img${img_cnt}/UNet_seed$i/final_model.pth" \
                    --train_dataset "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/" \
                    --val_dataset "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_test/" \
                    --direc "../results/$pct/${cancer}_img${img_cnt}/UNet_seed$i/" \
                    --batch_size 1 --modelname "UNet" --imgsize 200 --gray "no"

                    time python train_nnunet.py --train_dataset "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/" \
                    --val_dataset "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/" \
                    --direc "../results/$pct/${cancer}_img${img_cnt}/nnUNet_seed$i/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "nnUNet" \
                    --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

                    time python test_nnunet.py --loaddirec "../results/$pct/${cancer}_img${img_cnt}/nnUNet_seed$i/final_model.pth" \
                    --train_dataset "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_train/" \
                    --val_dataset "../../external_data/MoNuSeg/$pct/$cancer/img${img_cnt}_test/" \
                    --direc "../results/$pct/${cancer}_img${img_cnt}/nnUNet_seed$i/" \
                    --batch_size 1 --modelname "nnUNet" --imgsize 200 --gray "no"
                fi
            done
        done
    done
done


# for i in $(seq 1 3);
# do
#     for cancer in normal tumor;
#     do
#         for pct in GLySACByTumor_intraimage5pct_200x200 GLySACByTumor_intraimage20pct_200x200 GLySACByTumor_intraimage50pct_200x200;
#         do
#             # Iterate over possible files. Please change the range to the max number of files.
#             for img_cnt in $(seq 0 15);
#             do
#                 if [ -d "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/" ]; then
#                     time python train.py --train_dataset "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/" \
#                     --val_dataset "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/" \
#                     --direc "../results/$pct/${cancer}_img${img_cnt}/MedT_seed$i/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "MedT" \
#                     --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

#                     time python test.py --loaddirec "../results/$pct/${cancer}_img${img_cnt}/MedT_seed$i/final_model.pth" \
#                     --train_dataset "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/" \
#                     --val_dataset "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_test/" \
#                     --direc "../results/$pct/${cancer}_img${img_cnt}/MedT_seed$i/" \
#                     --batch_size 1 --modelname "MedT" --imgsize 200 --gray "no"

#                     time python train_unet.py --train_dataset "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/" \
#                     --val_dataset "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/" \
#                     --direc "../results/$pct/${cancer}_img${img_cnt}/UNet_seed$i/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "UNet" \
#                     --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

#                     time python test_unet.py --loaddirec "../results/$pct/${cancer}_img${img_cnt}/UNet_seed$i/final_model.pth" \
#                     --train_dataset "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/" \
#                     --val_dataset "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_test/" \
#                     --direc "../results/$pct/${cancer}_img${img_cnt}/UNet_seed$i/" \
#                     --batch_size 1 --modelname "UNet" --imgsize 200 --gray "no"

#                     time python train_nnunet.py --train_dataset "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/" \
#                     --val_dataset "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/" \
#                     --direc "../results/$pct/${cancer}_img${img_cnt}/nnUNet_seed$i/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "nnUNet" \
#                     --learning_rate 0.001 --imgsize 200 --gray "no" --seed $i

#                     time python test_nnunet.py --loaddirec "../results/$pct/${cancer}_img${img_cnt}/nnUNet_seed$i/final_model.pth" \
#                     --train_dataset "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_train/" \
#                     --val_dataset "../../external_data/GLySAC/$pct/$cancer/img${img_cnt}_test/" \
#                     --direc "../results/$pct/${cancer}_img${img_cnt}/nnUNet_seed$i/" \
#                     --batch_size 1 --modelname "nnUNet" --imgsize 200 --gray "no"
#                 fi
#             done
#         done
#     done
# done
