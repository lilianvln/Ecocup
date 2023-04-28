import os
import pickle
import cv2
import numpy as np
from skimage.transform import resize
from skimage.feature import hog
from skimage.color import rgb2gray


class Classifier:

    def __init__(self, model, window_size=(50, 30), stride=100, threshold=0.5):
        # Définir les tailles de fenêtres glissantes
        self.window_size = window_size
        # Définir le pas de la fenêtre
        self.stride = stride
        self.threshold = threshold
        self.model = model

    def iou(self, i1, j1, h1, l1, i2, j2, h2, l2):
        x_inter1 = max(j1, j2)
        y_inter1 = max(i1, i2)
        x_inter2 = min(j1 + l1, j2 + l2)
        y_inter2 = min(i1 + h1, i2 + h2)
        width_inter = max(0, x_inter2 - x_inter1)
        height_inter = max(0, y_inter2 - y_inter1)
        area_inter = width_inter * height_inter

        area_box1 = h1 * l1
        area_box2 = h2 * l2
        area_union = area_box1 + area_box2 - area_inter
        iou = area_inter / area_union
        return iou

    def filtre_nms(self, results_in, tresh_iou=0.5):
        # role : si chevauchement trop fort entre deux detections, garder celle dont la confiance est maximale
        results_out = np.empty((0, 8))  # initialiser un tableau de sortie vide
        unique_ids = np.unique(results_in[:, 0])
        for i in unique_ids:  # image par image
            results_in_i = results_in[results_in[:, 0] == i]
            # trier les boites par score de confiance decroissant
            results_in_i = results_in_i[results_in_i[:, 5].argsort()[::-1]]
            # liste des boites que l'on garde pour cette image a l'issue du NMS
            results_out_i = np.empty((0, 8))
            # on garde forcement la premiere boite, la plus sure
            results_out_i = np.vstack((results_out_i, results_in_i[0]))
            # pour toutes les boites suivantes, les comparer a celles que l'on garde
            for n in range(1, len(results_in_i)):
                for m in range(len(results_out_i)):
                    if self.iou(results_in_i[n, 1], results_in_i[n, 2], results_in_i[n, 3], results_in_i[n, 4],
                                results_out_i[m, 1], results_out_i[m, 2], results_out_i[m, 3],
                                results_out_i[m, 4]) > tresh_iou:
                        # recouvrement important ,
                        # et la boite de results_out_i a forcement un score plus haut
                        break
                    elif m == len(results_out_i) - 1:
                        # c'etait le dernier test pour verifier si cette detection est a conserver
                        results_out_i = np.vstack((results_out_i, results_in_i[n]))
                # ajouter les boites de cette image a la liste de toutes les boites
            results_out = np.vstack((results_out, results_out_i))
        return results_out

    def ajouter_cadre(self, numero_image, coordonnee_ligne, coordonne_colonne, hauteur, largeur, score, img_shape_0, img_shape_1):
        # Charger l'image
        image = cv2.imread('test/%03d.jpg'%numero_image)
        image = cv2.resize(image, (img_shape_1, img_shape_0))
        # Dessiner le cadre autour de l'objet
        cv2.rectangle(image, (coordonne_colonne, coordonnee_ligne), (coordonne_colonne + largeur, coordonnee_ligne + hauteur), (0, 255, 0), 2)

        # Ajouter le score à l'image
        cv2.putText(image, f'Score : {score}', (coordonne_colonne, coordonnee_ligne - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Retourner l'image modifiée
        cv2.imshow("Image avec cadre", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predict(self):
        # Step 1: Load dataset
        image_dir_test = 'test/'
        image_test = os.listdir(image_dir_test)
        predict = []
        for i in range(len(image_test)):
            img = image_test[i]
            if img != '.DS_Store':
                img = cv2.imread(image_dir_test + img)
                detections = np.empty((0, 8))
                scale_factor = 1.0
                while scale_factor > 0.01 and img.shape[0] > self.window_size[0] and img.shape[1] > self.window_size[1]:
                    for y in range(0, img.shape[0] - self.window_size[0], self.stride):
                        for x in range(0, img.shape[1] - self.window_size[1], self.stride):
                            # Extraire la région de l'image correspondant à la fenêtre courante
                            window = img[y:y + self.window_size[0], x:x + self.window_size[1], :]
                            # Redimensionner la région pour qu'elle ait la taille de la fenêtre glissante
                            resized_window = resize(window, self.window_size)
                            # Extraire les descripteurs HOG de la région
                            hog_features=[]
                            hog_features.append(hog(rgb2gray(resized_window), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)))   # extract HOG features
                            # Ajouter les descripteurs HOG à la liste des exemples négatifs
                            pred = self.model.predict_proba(hog_features)
                            score = pred[:, 1][0]
                            if score > self.threshold:
                                img_shape_0 = img.shape[0]
                                img_shape_1 = img.shape[1]
                                new_row = np.array([i, y / scale_factor, x / scale_factor, self.window_size[0] / scale_factor, self.window_size[1] / scale_factor,  score, img_shape_0, img_shape_1])
                                new_row = new_row.reshape(1, 8)
                                detections = np.append(detections, new_row, axis=0)
                    # Réduire la taille de l'image pour le prochain passage de la fenêtre glissante
                    scale_factor *= 0.8
                    img = resize(img, (int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor)))
                if detections.shape[0] > 1:
                    predict.append(self.filtre_nms(detections))
                elif detections.shape[0] == 1:
                    predict.append(detections)

        return predict

with open('model_HOG_50x30.p', 'rb') as f:
        model = pickle.load(f)
classifier = Classifier(model=model, stride=100, threshold=0.7)
predict = classifier.predict()

for i in predict:
    if i != []:
        print(i[0])
        numero_image = int(i[0][0])
        coordonnee_ligne = int(i[0][1])
        coordonne_colonne = int(i[0][2])
        hauteur = int(i[0][3])
        largeur = int(i[0][4])
        score = i[0][5]
        img_shape_0 = int(i[0][6])
        img_shape_1 = int(i[0][7])
        classifier.ajouter_cadre(numero_image, coordonnee_ligne, coordonne_colonne, hauteur, largeur, score, img_shape_0, img_shape_1)
