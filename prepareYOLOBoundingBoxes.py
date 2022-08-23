import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

def read_dataset_fromKaggle(link):

  os.system('mkdir ~/.kaggle')
  os.system('cp kaggle.json ~/.kaggle/')
  os.system('chmod 600 ~/.kaggle/kaggle.json')
  os.system(f'kaggle datasets download ${link}')

  dataset_name = link.split('/')[-1]
  zip_path = os.path.join(os.getcwd(),dataset_name+'.zip')
  os.system(f'unzip {zip_path}')

def convertToYoloBBox(bbox, size):
# This is taken from https://pjreddie.com/media/files/voc_label.py .
  dw = 1. / size[1]
  dh = 1. / size[0]
  x = (bbox[0] + bbox[1]) / 2.0
  y = (bbox[2] + bbox[3]) / 2.0
  w = bbox[1] - bbox[0]
  h = bbox[3] - bbox[2]
  x = x * dw
  w = w * dw
  y = y * dh
  h = h * dh
  return (x, y, w, h)





read_dataset_fromKaggle('klemenko/kitti-dataset')

os.system('mkdir ./data_object_label_2/training/label_2_darknet')
imgs_list = os.listdir('./data_object_image_2/training/image_2')

labels = ['Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' , 'DontCare']
for img_filename in imgs_list:
  id = img_filename.split('.')[0]
  data =  pd.read_csv(os.path.join('./data_object_label_2/training/label_2/',f'{id}.txt'), sep=" ", 
                  names=['label', 'truncated', 'occluded', 'alpha', 
                        'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 
                        'bbox_ymax', 'dim_height', 'dim_width', 'dim_length', 
                        'loc_x', 'loc_y', 'loc_z', 'rotation_y', 'score'])
  img = cv2.imread(os.path.join('./data_object_image_2/training/image_2',img_filename))
  f =  open(f'./data_object_label_2/training/label_2_darknet/{id}.txt','w+')
  txtfile = ''
  for i in range(len(data)):
    txtfile += str(labels.index(data.loc[i,['label']].values[0])) + ' '
    xmin = data.loc[i,['bbox_xmin']].values[0]
    xmax = data.loc[i,['bbox_xmax']].values[0]
    ymin = data.loc[i,['bbox_ymin']].values[0]
    ymax = data.loc[i,['bbox_ymax']].values[0]
    xc,yc,w,h = convertToYoloBBox([xmin,xmax,ymin,ymax],img.shape[:2])
    
    txtfile += str(xc) + ' '
    txtfile += str(yc) + ' '
    txtfile+= str(w) + ' '
    txtfile+= str(h)

    txtfile +='\n'

  f.write(txtfile)
  f.readline()
  f.close()