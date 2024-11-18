import os

# Dataset path
data_path = './data/raw'
train_path = f'{data_path}/seg_train/seg_train'
test_path = f'{data_path}/seg_test/seg_test'
pred_path = f'{data_path}/seg_pred/seg_pred'

# Image data properties
image_size = (150, 150)
img_height = 150
img_width = 150
img_channels = 3
batch_size = 32
class_names = [pathname for pathname in os.listdir(train_path) if '.' not in pathname]
