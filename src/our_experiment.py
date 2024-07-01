'''
This file is used to run the experiment on our method.

'''

import os
import sys
import time
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.log_util import log

config_path = 'config'
config_name = 'config'
@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def run(cfg: DictConfig) -> None:
    log(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Configs
    data_config = 'config/MoNuSeg_data.yaml'
    model_config = 'config/MoNuSeg_AIAE.yaml'
    reg2seg_cfg = 'config/MoNuSeg_reg2seg.yaml'

    # Run the experiment

    # Step 1: Preprocess the data
    log("================[Step 1] Preprocess the data================\n")
    patch_size = cfg.data.patch_size[0]
    aug_patch_size = cfg.data.aug_patch_size[0]
    preprocessed_data_path = f'../data/{cfg.data.train_name}_patch_{patch_size}x{patch_size}/'

    if os.path.exists(preprocessed_data_path):
        log(f"Preprocessed data already exists at {preprocessed_data_path}. \
                 Skipping preprocessing...")
    else:
        log(f"Preprocessing data to {preprocessed_data_path}...")
        os.system(f"python ./preprocessing/prepare_MoNuSeg.py \
                    --patch_size {patch_size} \
                    --aug_patch_size {aug_patch_size}")

    # Step 2:Subsample & Augment the data
    #./config/MoNuSeg_data.yaml will be created by augment_MoNuSeg.py
    log("================[Step 2] Sub-sampling & Augmenting the data================\n")
    percent = cfg.data.sample_percent
    organ_type = cfg.data.organ_type
    multiplier = cfg.data.aug_multiplier
    random_seed = cfg.data.random_seed
    detection = cfg.data.get('detection', 'None')
    aug_data_path = f'../data/{percent:.3f}_{organ_type}_m{multiplier}_{cfg.data.train_name}_augmented_patch_{aug_patch_size}x{aug_patch_size}/'

    log(f"Sub-sampling & Augmenting data to {aug_data_path}...")
    os.system(f"python ./preprocessing/augment_MoNuSeg.py \
                --patch_size {patch_size} \
                --augmented_patch_size {aug_patch_size} \
                --percentage {percent} \
                --multiplier {multiplier} \
                --organ {organ_type} \
                --detection {detection} \
                --random_seed {random_seed}")

    # Step 3: Train AIAE
    log("================[Step 3] Training AIAE================\n")
    os.system(f"python train_AIAE.py \
              --mode train \
              --data_config {data_config} \
              --model_config {model_config} \
              --num-workers {cfg.aiae.num_workers}")

    # Step 4: Generate paired matches using AIAE infer
    log("================[Step 4] Generating paired matches using AIAE infer================\n")
    os.system(f"python train_AIAE.py \
              --mode infer \
              --data_config {data_config} \
              --model_config {model_config} \
              --num-workers {cfg.aiae.num_workers}")

    # Step 5: Train Reg2Seg using paired matches
    log("================[Step 5] Training Reg2Seg using paired matches================\n")
    os.system(f"python train_reg2seg.py \
              --data_config {data_config} \
              --model_config {reg2seg_cfg} \
              --mode train")

    # Step 6: Infer & Evaluate Reg2Seg
    log("================[Step 6]Infering & Evaluating Reg2Seg================\n")
    os.system(f"python train_reg2seg.py \
              --data_config {data_config} \
              --model_config {reg2seg_cfg} \
              --mode infer")

    log("Experiment completed.")


if __name__ == "__main__":
    run()