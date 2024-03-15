# ThesisCode and data

## Chapter 5&6

### Setup

Please refer to Monodepth2.

### Training and Test
```shell
python train.py  --model_name 'model name' --log_dir 'log direction' --split sunny_UAV --dis_use_equ --data_path 'datasets' --dataset endo  --batch_size 5 --num_workers 8  --height 192  --width 320 --num_epoch 20 --JPEG 
python test_simple.py --model_name 'model name' --image_path 'image name'
```
For more details please refer to 'options.py'


## Chapter 7 point data

The point data was collected by Kinectv2 and downsampling, and in the folder 'Chapter7Data'
