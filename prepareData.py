import os

def read_dataset_fromKaggle(link):

  os.system('mkdir ~/.kaggle')
  os.system('cp kaggle.json ~/.kaggle/')
  os.system('chmod 600 ~/.kaggle/kaggle.json')
  os.system(f'kaggle datasets download ${link}')

  dataset_name = link.split('/')[-1]
  zip_path = os.path.join(os.getcwd(),dataset_name+'.zip')
  os.system(f'unzip {zip_path}')


def main():
    read_dataset_fromKaggle('klemenko/kitti-dataset')

main()