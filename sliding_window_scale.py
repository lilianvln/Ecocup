import cv2
import pickle
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize

def sliding_window_scale(image_path, window_size, stride, num_scales):
    # Charger le modèle
    with open('model.p', 'rb') as f:
        model = pickle.load(f)
    # Charger l'image
    img = cv2.imread(image_path)
    # Définir les facteurs d'agrandissement
    scales = [2*x for x in range(1, num_scales)]
    # Parcourir chaque échelle
    for scale in scales:
        # Redimensionner l'image
        img_scaled = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
        # Parcourir l'image avec la fenêtre glissante
        for y in range(0, img_scaled.shape[0] - window_size[1], stride):
            for x in range(0, img_scaled.shape[1] - window_size[0], stride):
                # Extraire la sous-image
                window = img_scaled[y:y+window_size[1], x:x+window_size[0]]
                # Prédire la fenêtre
                features_img = []
                resize_window = resize(window, (30, 30))
                features = hog(resize_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), channel_axis=-1)
                features_img.append(features)
                probabilities = model.predict_proba(features_img)
                # Récupérer le score de la prédiction pour la classe positive
                score = probabilities[:, 1]
                # Afficher la sous-image si le score est supérieur à un seuil
                if score > 0.95:
                    cv2.imshow("Window", window)
                    cv2.waitKey(0)
