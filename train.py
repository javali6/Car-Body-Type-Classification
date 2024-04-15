import click
import utilities
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

@click.command()
@click.argument('data_path', required=True, type=str)
@click.argument('target_model_file', required=True, type=str)
@click.option('--epoch', type=int, help='Number of epochs', default=10)
@click.option('--test_size', type=float, help='Size of the test set', default=0.2)

def main(data_path, target_model_file, epoch, test_size):
    # if not epoch:
    #     epoch = 10
    # if not test_size:
    #     test_size = 0.0

    images, labels = utilities.loadData(data_path)
    print("Data is load")
    labels_encoded, labels_encoder = utilities.encodeLabel(labels)
    imagesProc = utilities.preprocessImages(images)
    X_train, X_test, y_train, y_test = train_test_split(imagesProc, labels, test_size=test_size, stratify=labels)
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))


    # Stworzenie kopii etykiet
    labels_general = np.array(y_train.copy())
    labels_small = np.array(y_train.copy())
    labels_big = np.array(y_train.copy())

    labels_general_test = np.array(y_test.copy())
    labels_small_test = np.array(y_test.copy())
    labels_big_test = np.array(y_test.copy())

    # Zamiana określonych etykiet na 'small'
    labels_general[np.isin(labels_general, ['Coupe', 'Sedan', 'Convertible'])] = 'small'

    # Zamiana pozostałych etykiet na 'big'
    labels_general[np.isin(labels_general, ['SUV', 'Van'])] = 'big'

    labels_small[np.isin(labels_general, ['big'])] = None
    labels_big[np.isin(labels_general, ['small'])] = None

    labels_general_test[np.isin(labels_general_test, ['Coupe', 'Sedan', 'Convertible'])] = 'small'

    # Zamiana pozostałych etykiet na 'big'
    labels_general_test[np.isin(labels_general_test, ['SUV', 'Van'])] = 'big'

    labels_small_test[np.isin(labels_general_test, ['big'])] = None
    labels_big_test[np.isin(labels_general_test, ['small'])] = None

    labels_encoded_small, labels_encoder_small = utilities.encodeLabel(labels_small)
    y_labels_small = to_categorical(labels_encoded_small, num_classes=4)

    labels_encoded_small_test = labels_encoder_small.transform(labels_small_test)
    y_labels_small_test = to_categorical(labels_encoded_small_test, num_classes=4)


    labels_encoded_big, labels_encoder_big = utilities.encodeLabel(labels_big)
    y_labels_big = to_categorical(labels_encoded_big, num_classes=3)

    labels_encoded_big_test = labels_encoder_big.transform(labels_big_test)
    y_labels_big_test = to_categorical(labels_encoded_big_test, num_classes=3)


    labels_encoded_general, labels_encoder_general = utilities.encodeLabel(labels_general)
    y_labels_general = to_categorical(labels_encoded_general, num_classes=2)

    labels_encoded_general_test = labels_encoder_general.transform(labels_general_test)
    y_labels_general_test = to_categorical(labels_encoded_general_test, num_classes=2)


    labels_encoded_finale, labels_encoder_finale = utilities.encodeLabel(y_train)
    y_labels_finale = to_categorical(labels_encoded_finale, num_classes=5)

    labels_encoded_finale_test = labels_encoder_finale.transform(y_test)
    y_labels_finale_test = to_categorical(labels_encoded_finale_test, num_classes=5)

    input_shape = X_train.shape[1:]
    # Definicja wejścia
    input_layer = Input(shape=input_shape, name='input_images')
    # Pierwsza warstwa konwolucyjna dla ogólnego poziomu (samochody małe/duże)
    conv_general1 = Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
    max_pool_general1 = MaxPooling2D((2, 2))(conv_general1)
    # Druga warstwa konwolucyjna dla ogólnego poziomu
    conv_general2 = Conv2D(64, (3, 3), activation='relu', padding='same')(max_pool_general1)
    max_pool_general2 = MaxPooling2D((2, 2))(conv_general2)
    # Warstwa spłaszczająca
    flatten_general = Flatten()(max_pool_general2)
    # Warstwa gęsta dla ogólnego poziomu
    dense_general = Dense(128, activation='relu')(flatten_general)
    # Warstwa wyjściowa dla ogólnego poziomu
    output_general = Dense(2, activation='softmax', name='output_general')(dense_general)


    # Pierwsza warstwa konwolucyjna dla szczegółowego poziomu (samochody małe)
    conv_specific1_small = Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
    max_pool_specific1_small = MaxPooling2D((2, 2))(conv_specific1_small)
    # Druga warstwa konwolucyjna dla szczegółowego poziomu
    conv_specific2_small = Conv2D(64, (3, 3), activation='relu', padding='same')(max_pool_specific1_small)
    max_pool_specific2_small = MaxPooling2D((2, 2))(conv_specific2_small)
    # Warstwa spłaszczająca
    flatten_specific_small = Flatten()(max_pool_specific2_small)
    merged_general_small = concatenate([dense_general, flatten_specific_small])
    # Warstwa gęsta dla szczegółowego poziomu
    dense_specific_small = Dense(128, activation='relu')(merged_general_small)
    # Warstwa wyjściowa dla szczegółowego poziomu (samochody małe)
    output_specific_small = Dense(4, activation='softmax', name='output_specific_small')(dense_specific_small)

    # Pierwsza warstwa konwolucyjna dla szczegółowego poziomu (samochody duże)
    conv_specific1_large = Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
    max_pool_specific1_large = MaxPooling2D((2, 2))(conv_specific1_large)
    # Druga warstwa konwolucyjna dla szczegółowego poziomu
    conv_specific2_large = Conv2D(64, (3, 3), activation='relu', padding='same')(max_pool_specific1_large)
    max_pool_specific2_large = MaxPooling2D((2, 2))(conv_specific2_large)
    # Warstwa spłaszczająca
    flatten_specific_large = Flatten()(max_pool_specific2_large)
    merged_general_large = concatenate([dense_general, flatten_specific_large])
    # Warstwa gęsta dla szczegółowego poziomu
    dense_specific_large = Dense(128, activation='relu')(merged_general_large)
    # Warstwa wyjściowa dla szczegółowego poziomu (samochody duże)
    output_specific_large = Dense(3, activation='softmax', name='output_specific_large')(dense_specific_large)

    # Połączenie obu wyjść
    merged = concatenate([dense_general, dense_specific_small, dense_specific_large])
    # Warstwa gęsta dla końcowego wyjścia
    final_dense = Dense(32, activation='relu')(merged)
    # Warstwa wyjściowa końcowa
    output_layer = Dense(5, activation='softmax', name='output_final')(final_dense)
    # Model dla samochodów
    model_cars = Model(inputs=input_layer, outputs=[output_general, output_specific_small, output_specific_large, output_layer])

    # Skompiluj model z odpowiednimi funkcjami straty dla każdej kategorii
    model_cars.compile(optimizer='adam',
                        loss={'output_general': 'sparse_categorical_crossentropy',
                            'output_specific_small': 'sparse_categorical_crossentropy',
                            'output_specific_large': 'sparse_categorical_crossentropy',
                            'output_final': 'sparse_categorical_crossentropy'},
                        metrics=['accuracy'],
                        )


    def custom_data_generator2(X_train, y_train, encoded_finale, encoded_small, encoded_big, encoded_general,batch_size=16):
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='constant',
            cval=0.5
        )

        size = len(y_train)
        idx = list(range(size))
        generator = datagen.flow(X_train, idx, batch_size=batch_size)

        while True:
            batch = generator.next()
            batch_inputs = batch[0]
            batch_outputs = {
                'output_general': encoded_general[batch[1]],
                'output_specific_small': encoded_small[batch[1]],
                'output_specific_large': encoded_big[batch[1]],
                'output_final': encoded_finale[batch[1]],
            }
            yield batch_inputs, batch_outputs


    train_generator2 = custom_data_generator2(X_train,y_train,encoded_finale=labels_encoded_finale,encoded_big=labels_encoded_big,encoded_general=labels_encoded_general,encoded_small=labels_encoded_small,batch_size=64)

    # Trenowanie modelu z użyciem generatora
    history = model_cars.fit_generator(
        train_generator2,
        steps_per_epoch=len(X_train)//64,
        epochs=epoch,
        validation_data=(X_test, {'output_general': labels_encoded_general_test, 'output_specific_small': labels_encoded_small_test, 'output_specific_large': labels_encoded_big_test, 'output_final': labels_encoded_finale_test}),
    )

    utilities.plotPrint(history)

    utilities.saveModel(target_model_file, model_cars)



# python .\train.py "data/main" "models/main" --test_size 0.1
if __name__ == "__main__":
    main()