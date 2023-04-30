# Import des bibliothèques
import os
import pickle
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize, rotate, rescale
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


# Répertoire contenant les données d'apprentissage
imput_dir = 'clf_data'
# Les deux catégories d'images que l'on veut classifier
categories = ['empty', 'not_empty']
# Liste des features (caractéristiques) et des labels (étiquettes) pour chaque image
data = []
labels = []


# Boucle pour chaque catégorie d'image
for category_idx, category in enumerate(categories):
    # Boucle pour chaque image de la catégorie
    for file in os.listdir(os.path.join(imput_dir, category)):
        if file != '.DS_Store':  # Exclure les fichiers cachés du système
            # Chemin d'accès à l'image
            img_path = os.path.join(imput_dir, category, file)
            # Charger l'image en niveaux de gris
            img = imread(img_path, as_gray=True)

            # Augmentation de données : Zoom
            for zoom_factor in [0.8, 1.2]:
                # Zoomer l'image
                zoomed_img = rescale(img, zoom_factor, anti_aliasing=True)
                # Redimensionner l'image zoomée à la taille 60x30
                zoomed_img = resize(zoomed_img, (60, 30))
                # Calculer les caractéristiques HOG (Histogramme de Gradients Orientés) pour l'image zoomée
                features = hog(zoomed_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
                # Ajouter les caractéristiques et l'étiquette (catégorie) à la liste de données
                data.append(features)
                labels.append(category_idx)

            # Augmentation de données : Rotation
            if category == 'not_empty':  # Seulement pour les images de la catégorie "not_empty"
                for angle in range(10, 360, 10):
                    # Rotation de l'image
                    rotated_img = rotate(img, angle, resize=False)

                    # Augmentation de données : Zoom
                    for zoom_factor in [0.8, 1.2]:
                        # Zoomer l'image
                        zoomed_img = rescale(rotated_img, zoom_factor, anti_aliasing=True)
                        # Redimensionner l'image zoomée à la taille 50x30
                        zoomed_img = resize(zoomed_img, (60, 30))
                        # Calculer les caractéristiques et l'étiquette (catégorie) à la liste de données
                        features = hog(zoomed_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
                        data.append(features)
                        labels.append(category_idx)

# Transformation des listes data et labels en np.array
data = np.asarray(data)
labels = np.asarray(labels)
print(data.shape)


# Séparation de data et labels en deux np.array, un pour entraîner le modèle un second pour le tester
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


# Entrainement du classifieur
classifier = SVC(kernel='poly', C=10, gamma='scale', probability=True)

classifier.fit(X_train, y_train)

# Test du classifieur sur les données de test
y_prediction = classifier.predict(X_test)


# Calcul du taux de réussite et de f1, sur le X_test
score = accuracy_score(y_prediction, y_test)
print(f'{score * 100}% of samples were correctly classified')

f1 = f1_score(y_test, y_prediction, average='macro')
print(f'f1 score macro = {f1}')

# Création d'un fichier avec le modèle afiin de pourvoir le réutilliser
pickle.dump(classifier, open('model_SVM.p', 'wb'))
