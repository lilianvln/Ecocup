# Import des bibliothèques
import os
import pickle
import cv2
import numpy as np
from skimage.feature import hog
import csv


# Création d'une classe Classifier
class Classifier:

    def __init__(self, model, window_size=(60, 30), stride=30, threshold=0.5):
        # Définir les tailles de fenêtres glissantes
        self.window_size = window_size
        # Définir le pas de la fenêtre
        self.stride = stride
        # Définir le seuil
        self.threshold = threshold
        # Définir le modèle
        self.model = model



    def filtre_nms(self, detections, overlap_threshold=0.5):
        # Tri des détections en fonction de leur score (de la plus grande à la plus petite)
        detections = detections[np.argsort(-detections[:, 5])]
        # Initialisation de la liste des détections conservées
        detections_finales = []
        while detections.shape[0] > 0:
            # Ajout de la détection ayant le score le plus élevé à la liste des détections conservées
            detections_finales.append(detections[0])
            # Si les détections restantes ont un chevauchement suffisant avec la détection courante,
            # on les supprime de la liste des détections restantes
            if detections.shape[0] > 1:
                # Calcul de l'IOU (Intersection Over Union) entre la détection courante et les autres détections
                iou = self.calcule_iou(detections[0], detections[1:])
                # Suppression des détections ayant un chevauchement supérieur à un seuil donné
                detections = detections[1:][iou < overlap_threshold]
            else:
                break
        return np.array(detections_finales)


    def calcule_iou(self, detection, detections):
        # Calcule la surface de chaque détection
        detection_area = detection[3] * detection[4]
        detections_area = detections[:, 3] * detections[:, 4]
        # Calcule les coordonnées de l'intersection entre la détection courante et les autres détections
        xA = np.maximum(detection[2], detections[:, 2])
        yA = np.maximum(detection[1], detections[:, 1])
        xB = np.minimum(detection[2] + detection[4], detections[:, 2] + detections[:, 4])
        yB = np.minimum(detection[1] + detection[3], detections[:, 1] + detections[:, 3])
        # Calcule la surface de l'intersection entre la détection courante et les autres détections
        interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
        # Calcule l'IOU (Intersection Over Union) entre la détection courante et les autres détections
        iou = interArea / (detection_area + detections_area - interArea)
        return iou


    def predict(self):
        # Charge le dataset
        image_dir_test = 'test/'
        image_test = os.listdir(image_dir_test)
        predict = []
        for i in range(len(image_test)):
            img = image_test[i]
            if img != '.DS_Store':
                img = cv2.imread(image_dir_test + img, cv2.IMREAD_GRAYSCALE)
                detections = np.empty((0, 6))
                scale_factor = 1.0
                while scale_factor >= 0.01 and img.shape[0] > self.window_size[0] and img.shape[1] > self.window_size[1]:
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
                                new_row = np.array([i, line/scale_factor, column/scale_factor, self.window_size[0] / scale_factor, self.window_size[1] / scale_factor,  score])
                                new_row = new_row.reshape(1, 6)
                                detections = np.append(detections, new_row, axis=0)
                    # Réduire la taille de l'image pour le prochain passage de la fenêtre glissante
                    scale_factor *= 0.95
                    height_img = int(img.shape[0] * scale_factor)
                    width_img = int(img.shape[1] * scale_factor)
                    img = cv2.resize(img, (width_img, height_img))
                if detections.shape[0] > 1:
                    predict.append(self.filtre_nms(detections))
                elif detections.shape[0] == 1:
                    predict.append(detections)
        return predict


# Overture du fichier model.p pour l'utiliser dzan sle code
with open('model_SVM.p', 'rb') as f:
        model = pickle.load(f)


# Appel de la focntion classifier
classifier = Classifier(model=model, stride=25, threshold=0.8)
predict = classifier.predict()


# Sauvegarde des résultats dans un fichier CSV
def save_to_csv(predict, csv_file):
    with open(f"{csv_file}", 'w') as file:
        writer = csv.writer(file)
        for line in predict:
            numero_image = int(line[0][0])
            coordonnee_line = int(line[0][1])
            coordonnee_column = int(line[0][2])
            height_window = int(line[0][3])
            width_window = int(line[0][4])
            score = line[0][5]
            res = [numero_image, coordonnee_line, coordonnee_column, height_window, width_window, score]
            writer.writerow(res)

save_to_csv(predict, 'result.csv')
