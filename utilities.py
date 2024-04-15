import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def saveModel(model_dir, model):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(f'{model_dir}/model.h5')
    print(f"Model został zapisany pod nazwą: {model_dir}")

def loadModel(model_dir):
    loaded_model = tf.keras.models.load_model(f'{model_dir}/model.h5')
    print(f"Model został wczytany z pliku: {model_dir}")
    return loaded_model

def loadData(imageDir):
    csv_file = f'{imageDir}/labels/labels.csv'
    img_folder = f'{imageDir}/values'
    data_csv = pd.read_csv(csv_file)
    images_tab = []
    labels_string = []
    for index, row in data_csv.iterrows():
        filename = row['file_name']
        category = row['car_type']

        image_path = os.path.join(img_folder, filename)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images_tab.append(image)
        labels_string.append(category)
    images = np.array(images_tab)
    return (images,np.array(labels_string))

def plotPrint(history):
    print(history.history)
    epochs = np.arange(len(history.history['val_loss'])) + 1
    fig = plt.figure(figsize=(8, 4))
    if 'accuracy' in history.history:
        ax1 = fig.add_subplot(121)
        ax1.plot(epochs, history.history['loss'], c='b', label='Train loss')
        ax1.plot(epochs, history.history['val_loss'], c='g', label='Valid loss')
        plt.legend(loc='lower left');
        plt.grid(True)

        ax1 = fig.add_subplot(122)
        ax1.plot(epochs, history.history['accuracy'], c='b', label='Train acc')
        ax1.plot(epochs, history.history['val_accuracy'], c='g', label='Valid acc')
        plt.legend(loc='lower right');
        plt.grid(True)


    else:        
        ax1 = fig.add_subplot(121)
        ax1.plot(epochs, history.history['loss'], c='b', label='Train loss')
        ax1.plot(epochs, history.history['val_loss'], c='g', label='Valid loss')
        plt.legend(loc='lower left');
        plt.grid(True)
        if 'output_final_accuracy' in history.history:
            print('output_final_accuracy')
            ax1 = fig.add_subplot(122)
            ax1.plot(epochs, history.history['output_final_accuracy'], c='b', label='Train acc')
            ax1.plot(epochs, history.history['val_output_final_accuracy'], c='g', label='Valid acc')
            plt.legend(loc='lower right');
            plt.grid(True)
    plt.show()

def encodeLabel(labels_s):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_s)
    return (labels_encoded, label_encoder)

def preprocessImages(images):
    preprocessed_images = []
    for image in images:
        equalized_image = cv2.equalizeHist(np.uint8(image))
        normalized_image = equalized_image / 255.0
        preprocessed_images.append(normalized_image)
    preprocessed_images = np.array(preprocessed_images)
    return preprocessed_images