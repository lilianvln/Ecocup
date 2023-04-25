import cv2
import numpy as np
import pandas as pd
import os
from skimage.transform import resize
from sklearn.linear_model import LogisticRegression
from skimage.feature import hog
from skimage.color import rgb2gray


class Classifier:
    
    def __init__(self, window_size=(64, 64), stride = 100, threshold=0.5):
        # Initialiser le classificateur
        self.clf = LogisticRegression(max_iter=1000)
         # Définir les tailles de fenêtres glissantes
        self.window_size=window_size
        # Définir le pas de la fenêtre
        self.stride= stride
        self.threshold=threshold
    

    def fit(self):
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
                    resized_region = resize(region, self.window_size)
                    # Extraire les descripteurs HOG de la région
                    hog_features = hog(rgb2gray(resized_region), orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(3, 3), block_norm='L2-Hys', feature_vector=True)
                    positive_examples.append(hog_features)
            else :
                img=cv2.imread(image_dir_neg+img)
                scale_factor = 1.0
                while scale_factor>0.1 and img.shape[0] > self.window_size[0] and img.shape[1] > self.window_size[1]:
                    for y in range(0, img.shape[0] - self.window_size[0], self.stride):
                        for x in range(0, img.shape[1] - self.window_size[1], self.stride):
                            # Extraire la région de l'image correspondant à la fenêtre courante
                            window = img[y:y+self.window_size[0], x:x+self.window_size[1], :]
                            # Redimensionner la région pour qu'elle ait la taille de la fenêtre glissante
                            resized_window = resize(window, self.window_size)
                            # Extraire les descripteurs HOG de la région
                            hog_features = hog(rgb2gray(resized_window), orientations=9, pixels_per_cell=(8, 8),
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
        self.clf.fit(X, y)
        return self

    def iou(self, i1, j1, h1, l1,i2, j2, h2, l2):
        x_inter1 = max(j1, j2)
        y_inter1 = max(i1, i2)
        x_inter2 = min(j1+l1, j2+l2)
        y_inter2 = min(i1+h1, i2+h2)
        width_inter = max(0,x_inter2 - x_inter1)
        height_inter = max(0,y_inter2 - y_inter1)
        area_inter = width_inter * height_inter

        area_box1 = h1 * l1
        area_box2 = h2 * l2
        area_union = area_box1 + area_box2 - area_inter
        iou = area_inter / area_union
        return iou

    def filtre_nms(self, results_in , tresh_iou = 0.5):
        # role : si chevauchement trop fort entre deux detections, garder celle dont la confiance est maximale
        results_out = np.empty((0,6)) # initialiser un tableau de sortie vide 
        unique_ids = np.unique(results_in[:,0])
        for i in unique_ids: # image par image
            results_in_i = results_in[results_in[:,0] == i]
            # trier les boites par score de confiance decroissant
            results_in_i = results_in_i[ results_in_i[:,5].argsort()[::-1] ]
            # liste des boites que l'on garde pour cette image a l'issue du NMS
            results_out_i = np.empty((0,6))
            # on garde forcement la premiere boite, la plus sure
            results_out_i = np.vstack((results_out_i, results_in_i[0]))
            # pour toutes les boites suivantes, les comparer a celles que l'on garde
            for n in range(1,len(results_in_i)):
                for m in range(len(results_out_i)):
                    if self.iou(results_in_i[n,1],results_in_i[n,2],results_in_i[n,3],results_in_i[n,4], results_out_i[m,1],results_out_i[m,2],results_out_i[m,3],results_out_i[m,4]) > tresh_iou:
                    # recouvrement important ,
                    # et la boite de results_out_i a forcement un score plus haut
                        break
                    elif m == len(results_out_i)-1:
                    # c'etait le dernier test pour verifier si cette detection est a conserver
                        results_out_i = np.vstack((results_out_i , results_in_i[n]))
                # ajouter les boites de cette image a la liste de toutes les boites
            results_out = np.vstack((results_out ,results_out_i))
        return results_out

    def predict(self):
        # Step 1: Load dataset and contour coordinates
        image_dir_test = 'test/'
        image_test = os.listdir(image_dir_test)
        predict=[]
        for i in range(len(image_test)):
            img = image_test[i]
            img=cv2.imread(image_dir_test+img)
            detections=np.empty((0, 6))
            scale_factor = 1.0
            while scale_factor>0.1 and img.shape[0] > self.window_size[0] and img.shape[1] > self.window_size[1]:
                for y in range(0, img.shape[0] - self.window_size[0], self.stride):
                    for x in range(0, img.shape[1] - self.window_size[1], self.stride):
                        # Extraire la région de l'image correspondant à la fenêtre courante
                        window = img[y:y+self.window_size[0], x:x+self.window_size[1], :]
                        # Redimensionner la région pour qu'elle ait la taille de la fenêtre glissante
                        resized_window = resize(window, self.window_size)
                        # Extraire les descripteurs HOG de la région
                        hog_features = hog(rgb2gray(resized_window), orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(3, 3), block_norm='L2-Hys', feature_vector=True)
                                        # Ajouter les descripteurs HOG à la liste des exemples négatifs
                        pred = self.clf.predict_proba(hog_features.reshape(1,-1))[0][1]
                        if pred > self.threshold:
                            new_row = np.array([i, y/scale_factor, x/scale_factor, self.window_size[0]/scale_factor, self.window_size[1]/scale_factor, pred])
                            new_row = new_row.reshape(1, 6)
                            print("new")
                            print(new_row)
                            detections = np.append(detections, new_row,axis =0)

                # Réduire la taille de l'image pour le prochain passage de la fenêtre glissante
                scale_factor *= 0.8
                img = resize(img, (int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor)))
            if detections.shape[0] > 1:
                print("test")
                print(detections)
                predict.append(self.filtre_nms(detections))
            elif detections.shape[0] == 1 :
                predict.append(detections)
        return predict
