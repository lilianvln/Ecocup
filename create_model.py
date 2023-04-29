import os
import pickle
import cv2
import numpy as np
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
                    if self.iou(results_in_i[n, 1], results_in_i[n, 2], results_in_i[n, 3], results_in_i[n, 4], results_out_i[m, 1], results_out_i[m, 2], results_out_i[m, 3], results_out_i[m, 4]) > tresh_iou:
                        # recouvrement important, et la boite de results_out_i a forcément un score plus haut
                        break
                    elif m == len(results_out_i) - 1:
                        # c'etait le dernier test pour verifier si cette detection est à conserver
                        results_out_i = np.vstack((results_out_i, results_in_i[n]))
                # ajouter les boites de cette image a la liste de toutes les boites
            results_out = np.vstack((results_out, results_out_i))
        return results_out

    def ajouter_cadre(self, numero_image, coordonnee_column, coordonnee_line, height_window, width_window, score, height_img, witdh_img):
        # Charger l'image
        image = cv2.imread('test/%03d.jpg' %numero_image)
        image = cv2.resize(image, (witdh_img, height_img), interpolation=cv2.INTER_AREA)
        # Dessiner le cadre autour de l'objet
        cv2.rectangle(image, (coordonnee_column, coordonnee_line), (coordonnee_column + width_window, coordonnee_line + height_window), (0, 255, 0), 2)

        # Ajouter le score à l'image
        cv2.putText(image, f'Score : {score}', (coordonnee_column, coordonnee_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Retourner l'image modifiée
        cv2.imshow(f"{image.shape}", image)
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
                img = cv2.imread(image_dir_test + img, cv2.IMREAD_GRAYSCALE)
                detections = np.empty((0, 8))
                scale_factor = 1.0
                while scale_factor >= 0.5 and img.shape[0] > self.window_size[0] and img.shape[1] > self.window_size[1]:
                    height_img, width_img = img.shape
                    for line in range(0, height_img - self.window_size[0], self.stride):
                        for column in range(0, width_img - self.window_size[1], self.stride):
                            # Extraire la région de l'image correspondant à la fenêtre courante
                            window = img[line:line + self.window_size[0], column:column + self.window_size[1]]
                            # Redimensionner la région pour qu'elle ait la taille de la fenêtre glissante
                            resized_window = cv2.resize(window, (self.window_size[1], self.window_size[0]), interpolation=cv2.INTER_AREA)
                            # Extraire les descripteurs HOG de la région
                            hog_features=[]
                            hog_features.append(hog(resized_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)))   # extract HOG features
                            # Ajouter les descripteurs HOG à la liste des exemples négatifs
                            pred = self.model.predict_proba(hog_features)
                            score = pred[:, 1][0]
                            if score > self.threshold:
                                new_row = np.array([i, line, column, self.window_size[0], self.window_size[1],  score, height_img, width_img])
                                new_row = new_row.reshape(1, 8)
                                detections = np.append(detections, new_row, axis=0)
                    # Réduire la taille de l'image pour le prochain passage de la fenêtre glissante
                    scale_factor *= 0.8
                    height_img = int(img.shape[0] * scale_factor)
                    width_img = int(img.shape[1] * scale_factor)
                    img = cv2.resize(img, (width_img, height_img))
                if detections.shape[0] > 1:
                    predict.append(self.filtre_nms(detections))
                elif detections.shape[0] == 1:
                    predict.append(detections)
        return predict

with open('model_HOG_50x30.p', 'rb') as f:
        model = pickle.load(f)
classifier = Classifier(model=model, stride=30, threshold=0.7)
predict = classifier.predict()

for i in predict:
    if i != []:
        print(i[0])
        numero_image = int(i[0][0])
        coordonnee_line = int(i[0][1])
        coordonne_colonne = int(i[0][2])
        height_window = int(i[0][3])
        width_window = int(i[0][4])
        score = i[0][5]
        height_img = int(i[0][6])
        width_img = int(i[0][7])
        classifier.ajouter_cadre(numero_image, coordonnee_line, coordonne_colonne, height_window, width_window, score, height_img, width_img)
