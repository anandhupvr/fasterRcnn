import os
import glob
from PIL import Image
import numpy as np

all_items = []
dataset_path = 'dataset/'
def to_float(box):
    st = box.split(' ')
    return [float(i) for i in st]
# img_count = len(os.listdir(dataset_path+"images/human"))
file_annot = open('annotation.txt', 'w+')
# for i,j in zip(len(os.listdir("dataset/images")),os.listdir("dataset/images")):
# 	all_items[i] = glob.glob(dataset_path + "images/human/*.jpg")
cat = ['mushroom', 'spinach', 'tomato']
# cat = ['human']
j = 0
for i in os.listdir("dataset/images"):
	all_items.append(glob.glob("dataset/images/"+i + "/*.jpg"))
	j += 1
m = 0
for all_it in all_items:
	for i in all_it:
		image = Image.open(i)
		width, height = image.size
		# import pdb; pdb.set_trace()
		label_file = open(i.split("images")[0] + "labels"+ i.split("images")[-1].replace("jpg", "txt")).readlines()
		labels = np.array([[float(i)] for i in label_file if len(i) < 3])
		bb = np.array([to_float(i.strip()) for i in label_file if len(i) > 3], dtype=np.float32)
		lab =(i, bb[0][0], bb[0][1], bb[0][2], bb[0][3], cat[m])
		file_annot.write(str(lab)+"\n")
	m += 1