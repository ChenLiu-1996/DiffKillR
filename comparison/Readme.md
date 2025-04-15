### MedT

```
cd MedT

# Colon
python train.py --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/" \
--direc "../results/MoNuSegByCancer_200x200/Colon/MedT/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "MedT" \
--learning_rate 0.001 --imgsize 200 --gray "no"

python test.py --loaddirec "../results/MoNuSegByCancer_200x200/Colon/MedT/final_model.pth" \
--train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/test/" \
--direc "../results/MoNuSegByCancer_200x200/Colon/MedT/" \
--batch_size 1 --modelname "MedT" --imgsize 200 --gray "no"

# Breast
python train.py --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/" \
--direc "../results/MoNuSegByCancer_200x200/Breast/MedT/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "MedT" \
--learning_rate 0.001 --imgsize 200 --gray "no"

python test.py --loaddirec "../results/MoNuSegByCancer_200x200/Breast/MedT/final_model.pth" \
--train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/test/" \
--direc "../results/MoNuSegByCancer_200x200/Breast/MedT/" \
--batch_size 1 --modelname "MedT" --imgsize 200 --gray "no"

# Prostate
python train.py --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/" \
--direc "../results/MoNuSegByCancer_200x200/Prostate/MedT/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "MedT" \
--learning_rate 0.001 --imgsize 200 --gray "no"

python test.py --loaddirec "../results/MoNuSegByCancer_200x200/Prostate/MedT/final_model.pth" \
--train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/test/" \
--direc "../results/MoNuSegByCancer_200x200/Prostate/MedT/" \
--batch_size 1 --modelname "MedT" --imgsize 200 --gray "no"
```


### UNet

```
cd MedT

# Colon
python train_unet.py --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/" \
--direc "../results/MoNuSegByCancer_200x200/Colon/UNet/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "UNet" \
--learning_rate 0.001 --imgsize 200 --gray "no"

python test_unet.py --loaddirec "../results/MoNuSegByCancer_200x200/Colon/UNet/final_model.pth" \
--train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/test/" \
--direc "../results/MoNuSegByCancer_200x200/Colon/UNet/" \
--batch_size 1 --modelname "UNet" --imgsize 200 --gray "no"

# Breast
python train_unet.py --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/" \
--direc "../results/MoNuSegByCancer_200x200/Breast/UNet/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "UNet" \
--learning_rate 0.001 --imgsize 200 --gray "no"

python test_unet.py --loaddirec "../results/MoNuSegByCancer_200x200/Breast/UNet/final_model.pth" \
--train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/test/" \
--direc "../results/MoNuSegByCancer_200x200/Breast/UNet/" \
--batch_size 1 --modelname "UNet" --imgsize 200 --gray "no"

# Prostate
python train_unet.py --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/" \
--direc "../results/MoNuSegByCancer_200x200/Prostate/UNet/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "UNet" \
--learning_rate 0.001 --imgsize 200 --gray "no"

python test_unet.py --loaddirec "../results/MoNuSegByCancer_200x200/Prostate/UNet/final_model.pth" \
--train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/test/" \
--direc "../results/MoNuSegByCancer_200x200/Prostate/UNet/" \
--batch_size 1 --modelname "UNet" --imgsize 200 --gray "no"
```

### nnUNet

```
cd MedT

# Colon
python train_nnunet.py --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/" \
--direc "../results/MoNuSegByCancer_200x200/Colon/nnUNet/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "nnUNet" \
--learning_rate 0.001 --imgsize 200 --gray "no"

python test_nnunet.py --loaddirec "../results/MoNuSegByCancer_200x200/Colon/nnUNet/final_model.pth" \
--train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/test/" \
--direc "../results/MoNuSegByCancer_200x200/Colon/nnUNet/" \
--batch_size 1 --modelname "nnUNet" --imgsize 200 --gray "no"

# Breast
python train_nnunet.py --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/" \
--direc "../results/MoNuSegByCancer_200x200/Breast/nnUNet/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "nnUNet" \
--learning_rate 0.001 --imgsize 200 --gray "no"

python test_nnunet.py --loaddirec "../results/MoNuSegByCancer_200x200/Breast/nnUNet/final_model.pth" \
--train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/test/" \
--direc "../results/MoNuSegByCancer_200x200/Breast/nnUNet/" \
--batch_size 1 --modelname "nnUNet" --imgsize 200 --gray "no"

# Prostate
python train_nnunet.py --train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/" \
--direc "../results/MoNuSegByCancer_200x200/Prostate/nnUNet/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "nnUNet" \
--learning_rate 0.001 --imgsize 200 --gray "no"

python test_nnunet.py --loaddirec "../results/MoNuSegByCancer_200x200/Prostate/nnUNet/final_model.pth" \
--train_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/" \
--val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/test/" \
--direc "../results/MoNuSegByCancer_200x200/Prostate/nnUNet/" \
--batch_size 1 --modelname "nnUNet" --imgsize 200 --gray "no"
```


## PSM
```
cd PSM

python main_monuseg.py --mode 'train_base' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Colon --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/test/
python main_monuseg.py --mode 'generate_label' --method 'gradcam' --dataset_name MoNuSegByCancer_200x200/Colon --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/test/
python main_monuseg.py --mode 'train_second_stage' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Colon --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/test/
python main_monuseg.py --mode 'generate_voronoi' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Colon --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/test/
python main_monuseg.py --mode 'train_final_stage' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Colon --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/test/
python main_monuseg.py --mode 'test' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Colon --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/test/

python main_monuseg.py --mode 'train_base' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Breast --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/test/
python main_monuseg.py --mode 'generate_label' --method 'gradcam' --dataset_name MoNuSegByCancer_200x200/Breast --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/test/
python main_monuseg.py --mode 'train_second_stage' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Breast --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/test/
python main_monuseg.py --mode 'generate_voronoi' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Breast --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/test/
python main_monuseg.py --mode 'train_final_stage' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Breast --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/test/
python main_monuseg.py --mode 'test' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Breast --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/test/

python main_monuseg.py --mode 'train_base' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Prostate --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/test/
python main_monuseg.py --mode 'generate_label' --method 'gradcam' --dataset_name MoNuSegByCancer_200x200/Prostate --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/test/
python main_monuseg.py --mode 'train_second_stage' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Prostate --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/test/
python main_monuseg.py --mode 'generate_voronoi' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Prostate --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/test/
python main_monuseg.py --mode 'train_final_stage' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Prostate --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/test/
python main_monuseg.py --mode 'test' --crop_edge_size 200 --dataset_name MoNuSegByCancer_200x200/Prostate --data_train ../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/train/ --data_test ../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/test/

```

## LACSS
```
cd LACSS
python seg_monuseg.py
python seg_glysac.py
```


### SAM

```
cd SAM

# Colon
python test.py --val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Colon/test/" \
--direc "../results/MoNuSegByCancer_200x200/Colon/SAM/" --imgsize 200 --gray "no"

# Breast
python test.py --val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Breast/test/" \
--direc "../results/MoNuSegByCancer_200x200/Breast/SAM/" --imgsize 200 --gray "no"

# Prostate
python test.py --val_dataset "../../data/MoNuSeg/MoNuSegByCancer_200x200/Prostate/test/" \
--direc "../results/MoNuSegByCancer_200x200/Prostate/SAM/" --imgsize 200 --gray "no"
```