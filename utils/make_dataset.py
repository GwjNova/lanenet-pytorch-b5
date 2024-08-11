import os
from skimage import measure, color
import cv2
import numpy as np
from cli_helper import parse_args_dataset


def skimageFilter(gray):
    gray = (np.dstack((gray, gray, gray)) * 255).astype('uint8')
    labels = measure.label(gray[:, :, 0], connectivity=1)
    dst = color.label2rgb(labels, bg_label=0, bg_color=(0, 0, 0))
    gray = cv2.cvtColor(np.uint8(dst * 255), cv2.COLOR_RGB2GRAY)
    return gray


def exchange_to_instance(path, targetPath):
    # 读取二值化图片
    binary_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # # 使用中值滤波去除噪声点，同时保持图像边缘的清晰度
    median_filtered = cv2.medianBlur(binary_img, 3)

    instance = skimageFilter(median_filtered)
    cv2.imwrite(targetPath, instance)


def moveImageTodir(path):
    binary = os.path.join(path,"gt_image_binary")
    instance = os.path.join(path,"gt_image_instance")
    if os.path.isdir(binary):
        for file in os.listdir(binary):
            file_path = os.path.join(binary, file)
            object_path = os.path.join(instance, file)
            exchange_to_instance(file_path, object_path)


def dataTxt_created(data_dir,file_class):
    image_dir = os.path.join(data_dir, 'image')
    gt_instance_dir = os.path.join(data_dir, 'gt_image_instance')
    gt_binary_dir = os.path.join(data_dir, 'gt_image_binary')

    # 获取目录中的文件列表
    image_files = sorted(os.listdir(image_dir))
    gt_instance_files = sorted(os.listdir(gt_instance_dir))
    gt_binary_files = sorted(os.listdir(gt_binary_dir))

    # 写入train.txt文件
    txt_path = os.path.join(data_dir, file_class)
    with open(txt_path, 'w') as f:
        for img_file, bt_instance_file, bt_binary_file in zip(image_files, gt_instance_files, gt_binary_files):
            img_path = os.path.join(image_dir, img_file)
            gt_instance_path = os.path.join(gt_instance_dir, bt_instance_file)
            gt_binary_path = os.path.join(gt_binary_dir, bt_binary_file)
            f.write(f"{img_path} {gt_binary_path} {gt_instance_path}\n")
    print("创建完成")


def main():
    args = parse_args_dataset()

    for root, dirs, files in os.walk(args.path):
        for dir_name in dirs:
            if dir_name == 'img':
                old_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, 'image')
                os.rename(old_path, new_path)
            elif dir_name == 'label':
                old_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, 'gt_image_binary')
                os.rename(old_path, new_path)
        if not os.path.exists(os.path.join(root, 'gt_image_instance')):
            os.mkdir(os.path.join(root, 'gt_image_instance'))
    moveImageTodir(args.path)

    if args.classes == "val":
        dataTxt_created(args.path,"val.txt")
    elif args.classes == "train":
        dataTxt_created(args.path,"train.txt")


    

if __name__ == '__main__':
    main()