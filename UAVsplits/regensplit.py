# import random
# in_train_dir="/home/walle/Documents/monodepth2-modify/endosplits/train_files.txt"

# out_train_dir="/home/walle/Documents/monodepth2-modify/endosplits/23/train_files.txt"
# out_train_file = open(out_train_dir,'w',encoding='utf-8')  #输出文件位置

# in_val_dir="/home/walle/Documents/monodepth2-modify/endosplits/val_files.txt"

# out_val_dir="/home/walle/Documents/monodepth2-modify/endosplits/23/val_files.txt"
# out_val_file = open(out_val_dir,'w',encoding='utf-8')  #输出文件位置
 
# lines_train = []
 
# with open(in_train_dir, 'r',encoding='utf-8') as f:   #需要打乱的原文件位置
#     for line in f:  
#         lines_train.append(line)
# random.shuffle(lines_train)
 
# for line in lines_train:
#     out_train_file.write(line)
 
# out_train_file.close()

# lines_val = []

# with open(in_val_dir, 'r',encoding='utf-8') as f:   #需要打乱的原文件位置
#     for line in f:  
#         lines_val.append(line)
# random.shuffle(lines_val)
 
# for line in lines_val:
#     out_val_file.write(line)
 
# out_val_file.close()


import random
in_train_dir="/home/walle/Documents/monodepth2-modify/endosplits/train_files.txt"

out_train_dir="/home/walle/Documents/monodepth2-modify/endosplits/23/train_files.txt"
out_train_file = open(out_train_dir,'w',encoding='utf-8')  #输出文件位置

in_val_dir="/home/walle/Documents/monodepth2-modify/endosplits/val_files.txt"

out_val_dir="/home/walle/Documents/monodepth2-modify/endosplits/23/val_files.txt"
out_val_file = open(out_val_dir,'w',encoding='utf-8')  #输出文件位置
 
lines_train = []
 
with open(in_train_dir, 'r',encoding='utf-8') as f:   #需要打乱的原文件位置
    for line in f:  
        lines_train.append(line)
random.shuffle(lines_train)
 
for line in lines_train:
    out_train_file.write(line)
 
out_train_file.close()

lines_val = []

with open(in_val_dir, 'r',encoding='utf-8') as f:   #需要打乱的原文件位置
    for line in f:  
        lines_val.append(line)
random.shuffle(lines_val)
 
for line in lines_val:
    out_val_file.write(line)
 
out_val_file.close()

# import random
 
# out_file = open('/home/walle/Documents/monodepth2-modify/endosplits/23/','w',encoding='utf-8')  #输出文件位置
 
# lines = []
 
# with open('train_files.txt', 'r',encoding='utf-8') as f:   #需要打乱的原文件位置
#     for line in f:  
#         lines.append(line)
# random.shuffle(lines)
 
# for line in lines:
#     out_file.write(line)
 

# out_file.close()


