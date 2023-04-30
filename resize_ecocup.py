import os
import csv
from PIL import Image
import random as rd


image_dir_pos = 'train/images/pos'
image_dir_neg = 'train/images/neg'
csv_dir = 'train/labels_csv'
image_list_pos = os.listdir(image_dir_pos)
image_list_neg = os.listdir(image_dir_neg)
image_list = image_list_pos + image_list_neg
csv_list = os.listdir(csv_dir)


coordinates = {}
for csv_file in csv_list:
    if csv_file != '.DS_Store':
        image_name = csv_file.split('.')[0]
        with open(f"train/labels_csv/{csv_file}") as csvfile:
            reader = csv.reader(csvfile)
            row_image = []
            for row in reader:
                row_image.append(row)
        coordinates[image_name]=row_image


for csv_file in csv_list:
    if csv_file != '.DS_Store':
        image_name = csv_file.split('.')[0]
        image = Image.open(f"train/images/pos/{image_name}.jpg")
        for i_ in range(len(coordinates[image_name])):
            i, j, h, l, d = coordinates[image_name][i_]
            i = int(i)
            j = int(j)
            h = int(h)
            l = int(l)
            left = j
            top = i
            right = j + l
            bottom = i + h
            im1 = image.crop((left, top, right, bottom))
            im1.save(f'/Users/lilianvalin/Desktop/image_crop_pos/{image_name}_{i_}.png', 'png')


def create_train_neg (k):
    for csv_file in image_list_neg:
        if csv_file != '.DS_Store':
            image_name = csv_file.split('.')[0]
            image = Image.open(f"train/images/neg/{image_name}.jpg")
            width, height = image.size
            i = rd.randint(0, height)
            j = rd.randint(0, width)
            h = rd.randint(50, 300)
            l = rd.randint(50, 300)
            left = j
            top = i
            right = j + l
            bottom = i + h
            if left > 0 and top > 0 and right < width and bottom < height :
                im1 = image.crop((left, top, right, bottom))
                im1.save(f'image_crop_neg/{image_name}_{k}.jpg')

for k in range(20):
    create_train_neg(k)