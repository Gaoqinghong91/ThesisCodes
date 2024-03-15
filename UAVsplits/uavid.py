import random
#/home/walle/Disk_data/uavid/Germany/Germanyfull.txt
#/home/walle/Disk_data/uavid/China/Chinafull.txt

in_train_dir="/home/walle/Disk_data/uavid/Germany/Germanyfull.txt"
out_train_dir="/home/walle/Disk_data/uavid/Germany/Germanyfull_order.txt"
out_train_file = open(out_train_dir,'w',encoding='utf-8')  #输出文件位置


 
lines_train = []
 
with open(in_train_dir, 'r',encoding='utf-8') as f:   
    for line in f:  
        lines_train.append(line)
random.shuffle(lines_train)
 
for line in lines_train:
    out_train_file.write(line)
 
out_train_file.close()



