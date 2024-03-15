import os

from random import shuffle

from math import floor

def get_training_and_testing_sets(file_list):
    split = 0.7
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing
def randomize_files(file_list):
    shuffle(file_list)

def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    
    data_files = list(filter(lambda file: file.endswith('.png'), all_files))
    return data_files

file_list_train=get_file_list_from_dir('/home/walle/Documents/dataset/endoscopy/daVinci/train/image01')
randomize_files(file_list_train)
with open('train_files.txt','w') as f_train:    #设置文件对象
    for i in range(len(file_list_train)):
        out_train_l="daVinci/train"+" " + str(int(file_list_train[i].split(".")[0]))+" "+"l"+"\n"
        # print(out)   
        f_train.write(out_train_l)                 #将字符串写入文件中


file_list_val=get_file_list_from_dir('/home/walle/Documents/dataset/endoscopy/daVinci/val/image01')
randomize_files(file_list_val)
with open('val_files.txt','w') as f_train:    #设置文件对象
    for i in range(len(file_list_val)):
        out_val_l="daVinci/val"+" " + str(int(file_list_val[i].split(".")[0]))+" "+"l"+"\n"
        # print(out)   
        f_train.write(out_val_l)                 #将字符串写入文件中
      


# def get_training_and_testing_sets(file_list):
#     split = 0.7
#     split_index = floor(len(file_list) * split)
#     training = file_list[:split_index]
#     testing = file_list[split_index:]
#     return training, testing

# def randomize_files(file_list):
#     shuffle(file_list)

# def get_file_list_from_dir(datadir):
#     all_files = os.listdir(os.path.abspath(datadir))
    
#     data_files = list(filter(lambda file: file.endswith('.jpg'), all_files))
#     # print(data_files)
#     return data_files

# file_list=get_file_list_from_dir('/home/walle/Documents/dataset/endoscopy/daVinci/train/image01')
# randomize_files(file_list)

# train,val=get_training_and_testing_sets(file_list)
# with open('train_files.txt','w') as f_train:    #设置文件对象
#     for i in range(len(train)):
#         out_train_l="daVinci/train"+" " + str(int(train[i].split(".")[0]))+" "+"l"+"\n"
#         # print(out)   
#         f_train.write(out_train_l)                 #将字符串写入文件中

#         out_train_r="daVinci/train"+" " + str(int(train[i].split(".")[0]))+" "+"r"+"\n"
#         f_train.write(out_train_r) 


# with open('val_files.txt','w') as f_val:    #设置文件对象
#     for i in range(len(val)):
#         out_val_l="daVinci/train"+" " + str(int(val[i].split(".")[0]))+" "+"l"+"\n"
#         # print(out)   
#         f_val.write(out_val_l)                 #将字符串写入文件中

#         out_val_r="daVinci/train"+" " + str(int(val[i].split(".")[0]))+" "+"r"+"\n"
#         f_val.write(out_val_r) 



# file_list=get_file_list_from_dir('/home/walle/Documents/dataset/endoscopy/endoscopy_raw/rectified23/image01')
# randomize_files(file_list)

# train,val=get_training_and_testing_sets(file_list)
# with open('train_files.txt','w') as f_train:    #设置文件对象
#     for i in range(len(train)):
#         out_train_l="endoscopy_raw/rectified23"+" " + str(int(train[i].split(".")[0]))+" "+"l"+"\n"
#         # print(out)   
#         f_train.write(out_train_l)                 #将字符串写入文件中

#         out_train_r="endoscopy_raw/rectified23"+" " + str(int(train[i].split(".")[0]))+" "+"r"+"\n"
#         f_train.write(out_train_r) 


# with open('val_files.txt','w') as f_val:    #设置文件对象
#     for i in range(len(val)):
#         out_val_l="endoscopy_raw/rectified23"+" " + str(int(val[i].split(".")[0]))+" "+"l"+"\n"
#         # print(out)   
#         f_val.write(out_val_l)                 #将字符串写入文件中

#         out_val_r="endoscopy_raw/rectified23"+" " + str(int(val[i].split(".")[0]))+" "+"r"+"\n"
#         f_val.write(out_val_r) 



# def get_training_and_testing_sets(file_list):
#     split = 0.7
#     split_index = floor(len(file_list) * split)
#     training = file_list[:split_index]
#     testing = file_list[split_index:]
#     return training, testing

# def randomize_files(file_list):
#     shuffle(file_list)

# def get_file_list_from_dir(datadir):
#     all_files = os.listdir(os.path.abspath(datadir))
    
#     data_files = list(filter(lambda file: file.endswith('.jpg'), all_files))
#     # print(data_files)
#     return data_files

# file_list=get_file_list_from_dir('/home/walle/Documents/dataset/endoscopy/endoscopy_raw/rectified19/image01')
# randomize_files(file_list)

# train,val=get_training_and_testing_sets(file_list)
# with open('test_files23.txt','w') as f_train:    #设置文件对象
#     for i in range(len(train)):
#         out_train_l="endoscopy_raw/rectified23"+" " + str(int(train[i].split(".")[0]))+" "+"l"+"\n"
#         # print(out)   
#         f_train.write(out_train_l)                 #将字符串写入文件中

# file_list=get_file_list_from_dir('/home/walle/Documents/dataset/endoscopy/endoscopy_raw/rectified19/image01')
# randomize_files(file_list)

# train,val=get_training_and_testing_sets(file_list)
# with open('test_files19.txt','w') as f_train:    #设置文件对象
#     for i in range(len(train)):
#         out_train_l="endoscopy_raw/rectified19"+" " + str(int(train[i].split(".")[0]))+" "+"l"+"\n"
#         # print(out)   
#         f_train.write(out_train_l)                 #将字符串写入文件中



