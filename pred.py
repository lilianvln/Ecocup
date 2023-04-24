import cv2
import numpy as np
import pandas as pd
import os
from skimage.transform import resize
from sklearn.linear_model import LogisticRegression
from skimage.feature import hog
from skimage.color import rgb2gray

# Step 1: Load dataset and contour coordinates
image_dir_pos = 'train/images/pos/'
image_dir_neg = 'train/images/neg/'
csv_dir = 'train/labels_csv'
image_list_pos = os.listdir(image_dir_pos)
image_list_neg= os.listdir(image_dir_neg)
image_list=image_list_pos + image_list_neg
csv_list = os.listdir(csv_dir)

# Create a dictionary to store contour coordinates for each image
coordinates = {}
for csv_file in csv_list:
    image_num = csv_file.split('.')[0]+".jpg"
    df = pd.read_csv(os.path.join(csv_dir, csv_file), header=None)
    coordinates[image_num] = df.values

#print(coordinates.get('linsongt_pos_006.jpg')[0][0])


# Définir les tailles de fenêtres glissantes
window_size = (64, 64)

# Définir le pas de la fenêtre
stride = 100

# Initialiser le classificateur
classifier = LogisticRegression()

# Parcourir toutes les images avec les fenêtres glissantes 
positive_examples = []
negative_examples = []
for i in range(len(image_list)):
    img = image_list[i]
    boxes = coordinates.get(img)
    if boxes is not None and boxes.any():
        img=cv2.imread(image_dir_pos+img)
        print (img)
        print(boxes)
        for box in boxes:
            print(box)
            i, j, h, l = box[:4]
            # Extraire la région de l'image correspondant à la boîte de détection
            region = img[int(i):int(i)+int(h), int(j):int(j)+int(l), :]
            # Redimensionner la région pour qu'elle ait la taille de la fenêtre glissante
            resized_region = resize(region, window_size)
            # Extraire les descripteurs HOG de la région
            hog_features = hog(rgb2gray(resized_region), orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(3, 3), block_norm='L2-Hys', feature_vector=True)
            positive_examples.append(hog_features)
    else :
        img=cv2.imread(image_dir_neg+img)
        scale_factor = 1.0
        while scale_factor>0.1 and img.shape[0] > window_size[0] and img.shape[1] > window_size[1]:
            for y in range(0, img.shape[0] - window_size[0], stride):
                for x in range(0, img.shape[1] - window_size[1], stride):
                    # Extraire la région de l'image correspondant à la fenêtre courante
                    window = img[y:y+window_size[0], x:x+window_size[1], :]
                    # Redimensionner la région pour qu'elle ait la taille de la fenêtre glissante
                    resized_window = resize(window, window_size)
                    # Extraire les descripteurs HOG de la région
                    hog_features = hog(rgb2gray(resized_region), orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(3, 3), block_norm='L2-Hys', feature_vector=True)
                                    # Ajouter les descripteurs HOG à la liste des exemples négatifs
                    negative_examples.append(hog_features)
            # Réduire la taille de l'image pour le prochain passage de la fenêtre glissante
            scale_factor *= 0.8
            img = resize(img, (int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor)))

# Regrouper les exemples positifs et négatifs dans une seule matrice d'apprentissage X
X = np.vstack((positive_examples, negative_examples))

# Créer le vecteur de cibles y (1 pour les exemples positifs, 0 pour les exemples négatifs)
y = np.hstack((np.ones(len(positive_examples)), np.zeros(len(negative_examples))))

# Entraîner un classifieur de régression logistique sur les descripteurs HOG
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

