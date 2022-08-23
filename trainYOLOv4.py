import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os


def generateYoloRouter(imgs,router_filename,path):
    image_files = []

    for filename in imgs:
        if filename.endswith(".png"):
            image_files.append(path + filename)

    with open(router_filename, "w") as outfile:
        for image in image_files:
            outfile.write(image)
            outfile.write("\n")
        outfile.close()

list_imgs = os.listdir('./data_object_image_2/training/image_2')
train_imgs = list_imgs[:int(len(list_imgs)*0.8)]
val_imgs = list_imgs[int(len(list_imgs)*0.8):]

os.system('git clone https://github.com/AlexeyAB/darknet')

os.system("sed -i 's/OPENCV=0/OPENCV=1/' Makefile")
os.system("sed -i 's/GPU=0/GPU=1/' Makefile")
os.system("sed -i 's/CUDNN=0/CUDNN=1/' Makefile")
os.system("sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile")

os.system('make')

os.system('cp -rf ./obj.data ./darknet/data')

os.system('cp -rf ./obj.names ./darknet/data')


os.system('cp -rf ./yolov4-obj.cfg ./darknet/cfg')

os.system('mkdir ./darknet/data/test')

os.system('mkdir ./darknet/data/obj')

for train_img in train_imgs:
  idx = train_img.split('.')[0]
  os.system(f'cp ./data_object_image_2/training/image_2/{train_img} ./darknet/data/obj/')
  os.system(f'cp ./data_object_label_2/training/label_2_darknet/{idx}.txt ./darknet/data/obj/{idx}.txt')

for val_img in val_imgs:
  idx = val_img.split('.')[0]
  os.system(f'cp ./data_object_image_2/training/image_2/{val_img} ./darknet/data/test/')
  os.system(f'cp ./data_object_label_2/training/label_2_darknet/{idx}.txt ./darknet/data/test/{idx}.txt')


os.system('cp -rf ./data_object_label_2/training/label_2_darknet/* ./darknet/data/obj/')
os.system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137')
os.system('cd ./darknet/data')

generateYoloRouter(train_imgs,'train.txt','./darknet/data/obj/')
generateYoloRouter(val_imgs,'test.txt','./darknet/data/test/')

os.system('./darknet detector train data/obj.data ./darknet/cfg/yolov4-tiny-custom.cfg yolov4.conv.137 -dont_show -map')


