import os

from random import shuffle

from math import floor
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def get_training_and_testing_sets(file_list):
    split = 0.9
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing
def randomize_files(file_list):
    shuffle(file_list)

img_path=readlines("/home/walle/Disk_data/wildUAV/wild_img.txt")
randomize_files(img_path)
train,val= get_training_and_testing_sets(img_path)
#print(train[0]) 
#print(train[0].split(" ")[0]+" "+str(int(train[0].split(" ")[5]))+" "+"l"+"\n")##0 and 5 +" "+str(int(train[0].split(" ")[5]))+" "+"l"+"\n"

with open('/home/walle/Disk_data/wildUAV/train_files.txt','w') as f_train:    #设置文件对象
    for i in range(len(train)):
        out_train_l=train[i]+"\n"
        f_train.write(out_train_l)                 #将字符串写入文件中



with open('/home/walle/Disk_data/wildUAV/val_files.txt','w') as f_train:    #设置文件对象
    for i in range(len(val)):
        out_val_l=val[i]+"\n"   
        f_train.write(out_val_l)                 #将字符串写入文件中