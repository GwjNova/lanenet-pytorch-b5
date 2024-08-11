import os
import shutil
import sys

from skimage import measure, color

import cv2
import numpy as np


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


def moveImageTodir(path, targetPath):
    if os.path.isdir(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            object_path = os.path.join(targetPath, file)
            exchange_to_instance(file_path, object_path)
            print("{}转换完成".format(file))


if __name__ == '__main__':
    moveImageTodir(r"D:\python\data\dataset_B\test\gt_image_binary", r"D:\python\data\dataset_B\test\gt_image_instance")