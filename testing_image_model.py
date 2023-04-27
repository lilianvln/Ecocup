# Charger le modèle depuis le fichier "model.p"
with open('model.p', 'rb') as f:
    model = pickle.load(f)
img = []
img_path = '050.jpg'
image = imread(img_path)
image = resize(image, (30, 30))
features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), channel_axis=-1)
img.append(features)

# Transformer le tableau 2D de l'image en un tableau 1D
#image_array_1d = image_array.reshape(1, -1)

# Effectuer la prédiction en utilisant le modèle
prediction = model.predict(img)

# Afficher la prédiction
print(prediction)
