import torch
import os
import optuna
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, random_split, ConcatDataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from skimage.feature import local_binary_pattern

import cv2

def extract_lbp_features(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # Color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Single-channel image
        gray = image

    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 257), range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= np.sum(hist)
    return hist


parent_folder_path = 'images/cropped_dataset/'
categories = {'COVID': 0, 'Lung_Opacity': 1, 'Normal': 2, 'Viral_Pneumonia': 3}
arrays = []
category_amount = []

# get the category with the least images
for category in categories.keys():
    folder_path = os.path.join(parent_folder_path, category)
    image_files = os.listdir(folder_path)
    category_amount.append(len(image_files))

max_training = min(category_amount)

# convert the images into a pytorch dataset
for cat_folder, value in categories.items():

    folder_path = os.path.join(parent_folder_path, cat_folder)
    image_files = os.listdir(folder_path)

    for i, file_name in enumerate(image_files):

        if i >= max_training:
            break

        file_path = os.path.join(folder_path, file_name)
        image = Image.open(file_path)
        image_array = np.array(image)

        # verify all images are of the desired size
        if image.size != (250, 250):
            print(file_path, " IS NOT 250x250, it is: ", image.size)
            continue

        if image_array.shape != (250, 250):
            image_array = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

        # extract LBP features from image
        image_features = extract_lbp_features(image_array)

        arrays.append(image_features)

# reshape the array
arrays = np.array(arrays).astype(np.float32)
arrays = arrays / np.max(arrays)

arrays_labels = [0] * max_training
arrays_labels += [1] * max_training
arrays_labels += [2] * max_training
arrays_labels += [3] * max_training

arrays_labels = np.array(arrays_labels)

X_train, X_test, y_train, y_test = train_test_split(arrays, arrays_labels, test_size=0.2, random_state=42, stratify=arrays_labels)

y_train = to_categorical(y_train.astype(int), num_classes=4)
y_test = to_categorical(y_test.astype(int), num_classes=4)

# Modelo de red MLP personalizado
model_mlp = Sequential()
model_mlp.add(Dense(64, activation='relu', input_shape=(256,)))
model_mlp.add(Dropout(0.2))

model_mlp.add(Dense(32, activation='relu'))
model_mlp.add(Dropout(0.2))

model_mlp.add(Dense(4, activation='softmax'))

optimizer = Adam(learning_rate=0.001)
model_mlp.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model_mlp.fit(X_train, y_train, batch_size=128, epochs=50, verbose=1)

pred = model_mlp.predict(X_test)
print(pred)