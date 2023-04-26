import os
import pickle
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize, rotate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# prepare data
imput_dir = 'clf_data'
categories = ['empty', 'not_empty']

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(imput_dir, category)):
        if file != '.DS_Store':
            img_path = os.path.join(imput_dir, category, file)
            img = imread(img_path)
            img = resize(img, (30, 30))
            #data.append(img.flatten())
            features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), channel_axis=-1)  # extract HOG features
            data.append(features)
            labels.append(category_idx)

            # augment data with rotated versions of positive examples
            if category == 'not_empty':
                for angle in range(10, 360, 10):
                    rotated_img = rotate(img, angle, resize=False)
                    features = hog(rotated_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), channel_axis=-1)
                    data.append(features)
                    labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier

#classifier = SVC()
#parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

classifier = RandomForestClassifier()
parameters = {'n_estimators': [10, 50, 100, 200], 'random_state': [42, 100, 200, 500]}

# Recherche en grille des meilleurs paramètres
grid_search = GridSearchCV(classifier, parameters)


grid_search.fit(X_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(X_test)

score = accuracy_score(y_prediction, y_test)

print(f'{score * 100}% of samples were correctly classified')

pickle.dump(best_estimator, open('./model.p', 'wb'))

