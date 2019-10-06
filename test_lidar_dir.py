import os

path = '/home/pzr/kaggle/code3d/lidar/'
dir_list = os.listdir(path)

print (dir_list.count('host-a004_lidar1_1232833308300869606.bin'))

print (len(dir_list))
for dir in dir_list:
    print (dir + '\n')
print (dir_list[0])