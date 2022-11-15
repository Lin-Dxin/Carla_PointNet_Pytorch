import os
import shutil
import sys

# 警告
key = input("Warning:continue and it will move files which in current dir, continue?:(y/n)\n")
if key != 'y':
    print("shutdown!")
    sys.exit()
# 获取当前目录
source_add = os.getcwd()

# 获取当前目录下数据文件名
raw_data_list = os.listdir(source_add)
data_list = []
for file in raw_data_list:
    end = file[-3:-1]
    if end == 'np':
        data_list.append(file)
    



# 确认划分比例，并且将放入两个不通文件的数据的数据名存入列表
total_len = len(data_list)
test_offset = int(0.8 * total_len)
main_data_list = data_list[:test_offset]
test_file_list = data_list[test_offset:]


# 创建两个数据目录
test_file_name = 'TestData'
main_file_name = 'TrainAndValidateData'

if not os.path.exists(test_file_name):
    os.makedirs(test_file_name)
if not os.path.exists(main_file_name):
    os.makedirs(main_file_name)
    
# 移动数据
print('moving file.....')
for MainData in main_data_list:
    shutil.move(os.path.join(source_add, MainData), os.path.join(source_add,main_file_name,MainData))

for TestData in test_file_list:
    shutil.move(os.path.join(source_add, TestData), os.path.join(source_add,test_file_name,TestData))
print('complete!')
