import os
import pandas as pd

os.system('cd cfg')

os.system("sed -i 's/batch=64/batch=1/' yolov4-tiny-custom.cfg")
os.system("sed -i 's/subdivisions=32/subdivisions=1/' yolov4-tiny-custom.cfg")

os.system('cd ..')

os.system('./darknet detector valid data/obj.data cfg/yolov4-tiny-custom.cfg ./yolov4-tiny-custom.weights') #

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