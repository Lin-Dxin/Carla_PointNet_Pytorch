import os
import shutil
import sys

# 警告
key = input("Warning:continue and it will move files which in current dir, continue?:(y/n)\n")
if key != 'y':
    print("shuting down!")
    sys.exit()
# 获取当前目录
source_add = os.getcwd()

# 获取当前目录下数据文件名
raw_data_list = os.listdir(source_add)
data_list = []
for file in raw_data_list:
    end = file[-3:-1]
    if end == 'npy':
        data_list.append(file)

if len(data_list) == 0:
    print("No data in current dir!\nshuting down!")
    sys.exit()

# 确认数据划分号
partition_num = input("Type the sequence of this data(from 0 - 9):\n")
partition_num = int(partition_num)
if partition_num not in list(range(0,10)):
    print("Not a valid num!\n shuting down!\n")
    sys.exit()

# 确认放入validate的数据
total_len = len(data_list)
part_data_num = int(total_len / 10)
offset = part_data_num * partition_num
endpoint = offset + part_data_num
validate_data_list = data_list[offset : endpoint]
train_data_list = data_list[:offset] + data_list[endpoint:]

print(len(validate_data_list))
print(len(train_data_list))
# 创建两个数据目录
train_file_name = 'train'
validate_file_name = 'validate'

if not os.path.exists(train_file_name):
    os.makedirs(train_file_name)
if not os.path.exists(validate_file_name):
    os.makedirs(validate_file_name)

# 移动数据
print('moving file.....')
for TrainData in train_data_list:
    shutil.move(os.path.join(source_add, TrainData), os.path.join(source_add,train_file_name,TrainData))

for TestData in validate_data_list:
    shutil.move(os.path.join(source_add, TestData), os.path.join(source_add,validate_file_name,TestData))
print('complete!')