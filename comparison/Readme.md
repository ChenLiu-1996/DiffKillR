### MedT

```
cd MedT

python train.py --train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/train/" \
--val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/train/" \
--direc "../results/MoNuSegByCancer_200x200/colon/MedT/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "MedT" \
--learning_rate 0.001 --imgsize 200 --gray "no"

python test.py --loaddirec "../results/MoNuSegByCancer_200x200/colon/MedT/final_model.pth" \
--train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/train/" \
--val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/test/" \
--direc "../results/MoNuSegByCancer_200x200/colon/MedT/" \
--batch_size 1 --modelname "MedT" --imgsize 200 --gray "no"
```


### UNet

```
cd MedT

python train_unet.py --train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/train/" \
--val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/train/" \
--direc "../results/MoNuSegByCancer_200x200/colon/UNet/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "UNet" \
--learning_rate 0.001 --imgsize 200 --gray "no"

python test_unet.py --loaddirec "../results/MoNuSegByCancer_200x200/colon/UNet/final_model.pth" \
--train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/train/" \
--val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/test/" \
--direc "../results/MoNuSegByCancer_200x200/colon/UNet/" \
--batch_size 1 --modelname "UNet" --imgsize 200 --gray "no"
```

### nnUNet

```
cd MedT

python train_nnunet.py --train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/train/" \
--val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/train/" \
--direc "../results/MoNuSegByCancer_200x200/colon/nnUNet/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "nnUNet" \
--learning_rate 0.001 --imgsize 200 --gray "no"

python test_nnunet.py --loaddirec "../results/MoNuSegByCancer_200x200/colon/nnUNet/final_model.pth" \
--train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/train/" \
--val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/test/" \
--direc "../results/MoNuSegByCancer_200x200/colon/nnUNet/" \
--batch_size 1 --modelname "nnUNet" --imgsize 200 --gray "no"
```


## PSM
```
cd PSM

python main_monuseg.py --mode 'train_base' --crop_edge_size 200

python main_monuseg.py --mode 'generate_label' --method 'gradcam'

python main_monuseg.py --mode 'train_second_stage' --crop_edge_size 200

python main_monuseg.py --mode 'generate_voronoi' --crop_edge_size 200

python main_monuseg.py --mode 'train_final_stage' --crop_edge_size 200

python main_monuseg.py --mode 'test' --crop_edge_size 200

```