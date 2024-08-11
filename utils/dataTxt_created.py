# -*- coding: utf-8 -*-

import os

# 目录路径
data_dir = r'D:\python\data\dataset_B\train'
image_dir = os.path.join(data_dir, 'image')
gt_instance_dir = os.path.join(data_dir, 'gt_image_instance')
gt_binary_dir = os.path.join(data_dir, 'gt_image_binary')

# 获取目录中的文件列表
image_files = sorted(os.listdir(image_dir))
gt_instance_files = sorted(os.listdir(gt_instance_dir))
gt_binary_files = sorted(os.listdir(gt_binary_dir))

# 写入train.txt文件
with open('../data/dataset_B/train.txt', 'w') as f:
    for img_file, bt_instance_file, bt_binary_file in zip(image_files, gt_instance_files, gt_binary_files):
        img_path = os.path.join(image_dir, img_file)
        gt_instance_path = os.path.join(gt_instance_dir, bt_instance_file)
        gt_binary_path = os.path.join(gt_binary_dir, bt_binary_file)
        f.write(f"{img_path} {gt_binary_path} {gt_instance_path}\n")
