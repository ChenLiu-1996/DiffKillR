### MedT

```
cd MedT

python train.py --train_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/train/" --val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/train/" --direc "../results/MedT/" --batch_size 4 --epoch 400 --save_freq 50 --modelname "MedT" --learning_rate 0.00 --imgsize 200 --gray "no"

python test.py --loaddirec "./saved_model_path/model_name.pth" --val_dataset "../../external_data/MoNuSeg/MoNuSegByCancer_200x200/colon/test/" --direc "../results/MedT/" --batch_size 1 --modelname "MedT" --imgsize 200 --gray "no"

```
