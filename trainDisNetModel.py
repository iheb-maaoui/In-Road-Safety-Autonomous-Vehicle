import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras

d = {'Car':0,'Van':1,'Truck':2,'Pedestrian':3,'Person_sitting':4,'Cyclist':5,'Tram':6,'Misc':7,'DontCare':8}

list_imgs = os.listdir('./data_object_image_2/training/image_2')
val_imgs = list_imgs[int(len(list_imgs)*0.8):]

for i in val_imgs:
  if(i.endswith('.png')):
    img_indx = i.split('.')[0]
    f = open(f'./darknet/data/obj/{img_indx}_prediction.txt','w')
    f.close()
results_dir = os.listdir('./darknet/results')
for result_objs in list(d.keys()):
  df = pd.read_csv(f'./darknet/results/comp4_det_test_{result_objs}.txt',delimiter=' ',names=['img','score','x','y','w','h'])
  objects = df[df['score']>0.8]
  for img in pd.unique(objects['img']):
    img_objs = objects[objects['img']==img].loc[:,['score','x','y','w','h']].values
    img_objs[:,0] = int(d[result_objs])
    img_objs = img_objs.tolist()
    img_filename = str(img).zfill(6)
    f = open(f'./darknet/data/obj/{img_filename}_prediction.txt','a')
    lines = [f'{int(i[0])} {i[1]} {i[2]} {i[3]} {i[4]}\n' for i in img_objs]
    f.writelines(lines)
    f.close()

import numpy as np
train_imgs = list_imgs[:int(len(list_imgs)*0.8)]

train_data = pd.DataFrame(columns=['label','W', 'H', 'Diagonal','Avg_W','Avg_H','Avg_D', 'Distance'])
labels = {'Car':0,'Van':1,'Truck':2,'Pedestrian':3,'Person_sitting':4,'Cyclist':5,'Tram':6,'Misc':7,'DontCare':8}
for img_filename in train_imgs:
  img_indx = img_filename.split('.')[0]
  df = pd.read_csv(f'./data_object_label_2/training/label_2/{img_indx}.txt',delimiter=' ',names=['label', 'truncated', 'occluded', 'alpha', 
                        'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 
                        'bbox_ymax', 'dim_height', 'dim_width', 'dim_length', 
                        'loc_x', 'loc_y', 'loc_z', 'rotation_y', 'score'])
  df = df.iloc[:,[0,4,5,6,7,11,12,13]]
  df['W'] = (df['bbox_xmin'] + df['bbox_xmax'])/2.0
  df['H'] = (df['bbox_ymin'] + df['bbox_ymax'])/2.0
  df['Diagonal'] = np.sqrt(df['W'] ** 2 + df['H'] ** 2)
  df['Distance'] = np.absolute(df['loc_z'])#np.sqrt(df['loc_x']**2 + df['loc_y']**2 + df['loc_z']**2)
  df = df.loc[:,['label','W','H','Diagonal','Distance']]
  df['label'] = df['label'].apply(lambda col : labels[col])

  train_data = train_data.append(df,ignore_index=True)

val_data = pd.DataFrame(columns=['label','W', 'H', 'Diagonal','Avg_W','Avg_H','Avg_D', 'Distance'])
labels = {'Car':0,'Van':1,'Truck':2,'Pedestrian':3,'Person_sitting':4,'Cyclist':5,'Tram':6,'Misc':7,'DontCare':8}
for img_filename in val_imgs:
  img_indx = img_filename.split('.')[0]
  df = pd.read_csv(f'./data_object_label_2/training/label_2/{img_indx}.txt',delimiter=' ',names=['label', 'truncated', 'occluded', 'alpha', 
                        'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 
                        'bbox_ymax', 'dim_height', 'dim_width', 'dim_length', 
                        'loc_x', 'loc_y', 'loc_z', 'rotation_y', 'score'])
  df = df.iloc[:,[0,4,5,6,7,11,12,13]]
  df['W'] = (df['bbox_xmin'] + df['bbox_xmax'])/2.0
  df['H'] = (df['bbox_ymin'] + df['bbox_ymax'])/2.0
  df['Diagonal'] = np.sqrt(df['W'] ** 2 + df['H'] ** 2)
  df['Distance'] = np.absolute(df['loc_z']) #np.sqrt(df['loc_x']**2 + df['loc_y']**2 + df['loc_z']**2)
  df = df.loc[:,['label','W','H','Diagonal','Distance']]
  df['label'] = df['label'].apply(lambda col : labels[col])

  val_data = val_data.append(df,ignore_index=True)

width_means = [train_data[train_data['label'] == i]['W'].mean() for i in range(9)]
height_means = [train_data[train_data['label'] == i]['H'].mean() for i in range(9)]
diag_means = [train_data[train_data['label'] == i]['Diagonal'].mean() for i in range(9)]

for i in range(len(train_data)):
  train_data.loc[i,['Avg_W']] = width_means[train_data.loc[i,['label']].values[0]]
  train_data.loc[i,['Avg_H']] = height_means[train_data.loc[i,['label']].values[0]]
  train_data.loc[i,['Avg_D']] = diag_means[train_data.loc[i,['label']].values[0]]

width_means = [val_data[val_data['label'] == i]['W'].mean() for i in range(9)]
height_means = [val_data[val_data['label'] == i]['H'].mean() for i in range(9)]
diag_means = [val_data[val_data['label'] == i]['Diagonal'].mean() for i in range(9)]

for i in range(len(val_data)):
  val_data.loc[i,['Avg_W']] = width_means[val_data.loc[i,['label']].values[0]]
  val_data.loc[i,['Avg_H']] = height_means[val_data.loc[i,['label']].values[0]]
  val_data.loc[i,['Avg_D']] = diag_means[val_data.loc[i,['label']].values[0]]

#train_data.to_csv('./gdrive/MyDrive/train_distances2.csv')
#val_data.to_csv('./gdrive/MyDrive/val_distances2.csv')


train_data = pd.read_csv('./gdrive/MyDrive/train_distances2.csv')
val_data = pd.read_csv('./gdrive/MyDrive/val_distances2.csv')

train_data = train_data.iloc[:,1:]
val_data = val_data.iloc[:,1:]

from sklearn.preprocessing import MinMaxScaler
x_train = train_data.iloc[:,1:-1].values
y_train = train_data.iloc[:,-1].values.reshape(-1,1)
x_scalar = MinMaxScaler()
x_train = x_scalar.fit_transform(x_train)
y_scalar = MinMaxScaler()
y_train = y_scalar.fit_transform(y_train)

x_test = val_data.iloc[:,1:-1].values
y_test = val_data.iloc[:,-1].values.reshape(-1,1)
x_scalar = MinMaxScaler()
x_test = x_scalar.fit_transform(x_test)
y_scalar = MinMaxScaler()
y_test = y_scalar.fit_transform(y_test)

from DisNet import model

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=40,restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss',save_weights_only=True,save_best_only=True,filepath='./gdrive/MyDrive/DstNet.h5')
callbacks=[early_stopping,checkpoint]

history = model.fit(x_train,y_train,epochs=1000,validation_data=(x_test,y_test),shuffle=True,batch_size=512,callbacks=callbacks)

model.evaluate(x_test,y_test)