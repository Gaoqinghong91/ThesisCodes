# ThesisCode

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

## Setup

Please refer to Monodepth2.

## Training and Test
>python train.py  --model_name 'model name' --log_dir 'log direction' --split sunny_UAV --dis_use_equ --data_path 'datasets' --dataset endo  --batch_size 5 --num_workers 8  --height 192  --width 320 --num_epoch 20 --JPEG
>python test_simple.py --model_name 'model name' --image_path 'image name'
