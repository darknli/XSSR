from glob import glob
import os
import cv2


def check_dir(path):
    min_w = 1000
    min_h = 1000
    imgs = glob(os.path.join(path, '*'))
    for img in imgs:
        img = cv2.imread(img)
        print(img.shape)
        if img.shape[0] < min_w:
            min_w = img.shape[0]
        if img.shape[1] < min_h:
            min_h = img.shape[1]

    print(min_h, min_w)

check_dir(r'E:\Data\SR\DIV2K\DIV2K_train_LR_bicubic\X4')

