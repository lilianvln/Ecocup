import numpy as np
import pandas as pd
import os

# Step 1: Load dataset and contour coordinates
image_dir_pos = 'train/images/pos'
image_dir_neg = 'train/images/neg'
csv_dir = 'train/labels_csv'
image_list_pos = os.listdir(image_dir_pos)
image_list_neg= os.listdir(image_dir_neg)
image_list=image_list_pos + image_list_neg
csv_list = os.listdir(csv_dir)

# Create a dictionary to store contour coordinates for each image
coordinates = {}
for csv_file in csv_list:
    image_num = csv_file.split('.')[0]
    df = pd.read_csv(os.path.join(csv_dir, csv_file))
    coordinates[image_num] = df.values
