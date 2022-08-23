import os 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf

train_data = pd.read_csv('./gdrive/MyDrive/train_distances2.csv')
val_data = pd.read_csv('./gdrive/MyDrive/val_distances2.csv')


d = {'Car':0,'Van':1,'Truck':2,'Pedestrian':3,'Person_sitting':4,'Cyclist':5,'Tram':6,'Misc':7,'DontCare':8}
list_imgs = os.listdir('./data_object_image_2/training/image_2')
val_imgs = list_imgs[int(len(list_imgs)*0.8):]

for i in val_imgs:
  if(i.endswith('.png')):
    img_indx = i.split('.')[0]
    f = open(f'./darknet/data/test/{img_indx}_prediction.txt','w')
    f.close()
results_dir = os.listdir('./darknet/results')
for result_objs in list(d.keys()):
  df = pd.read_csv(f'./darknet/results/comp4_det_test_{result_objs}.txt',delimiter=' ',names=['img','score','x','y','w','h'])
  objects = df[df['score']>0.9]
  for img in pd.unique(objects['img']):
    img_objs = objects[objects['img']==img].loc[:,['score','x','y','w','h']].values
    img_objs[:,0] = int(d[result_objs])
    img_objs = img_objs.tolist()
    img_filename = str(img).zfill(6)
    f = open(f'./darknet/data/test/{img_filename}_prediction.txt','a')
    lines = [f'{int(i[0])} {i[1]} {i[2]} {i[3]} {i[4]}\n' for i in img_objs]
    f.writelines(lines)
    f.close()


img_data = pd.DataFrame(columns=['label','W', 'H', 'Diagonal','Avg_W','Avg_H','Avg_D'])

inference_data = pd.read_csv('./darknet/data/test/000025_prediction.txt',names = ['label','x','y','w','h'],sep=' ')

inference_data

img_data['label'] = inference_data['label']
img_data['W'] = inference_data['w']
img_data['H'] = inference_data['h']
img_data['Diagonal'] = np.sqrt(img_data['W'] **2 + img_data['H']**2)

width_means = [train_data[train_data['label'] == i]['W'].mean() for i in range(9)]
height_means = [train_data[train_data['label'] == i]['H'].mean() for i in range(9)]
diag_means = [train_data[train_data['label'] == i]['Diagonal'].mean() for i in range(9)]


for i in range(len(inference_data)):
  img_data.loc[i,['Avg_W']] = width_means[img_data.loc[i,['label']].values[0]]
  img_data.loc[i,['Avg_H']] = height_means[img_data.loc[i,['label']].values[0]]
  img_data.loc[i,['Avg_D']] = diag_means[img_data.loc[i,['label']].values[0]]

img_data

x_scalar = MinMaxScaler()

x_test = tf.convert_to_tensor(img_data.iloc[:,1:].values.astype(np.float32))

x_test = x_scalar.fit_transform(x_test)

from DisNet import model
ypred = model.predict(x_test)

y_scalar = MinMaxScaler()

y_scalar.fit_transform(train_data.iloc[:,2:-1].values)
y_scalar.inverse_transform(ypred).astype(np.int32)