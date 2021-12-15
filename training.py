import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPool2D, Reshape, Dense, Flatten
from sklearn.model_selection import KFold

from Prediction import Prediction


def load_data(source_path, cat_num, limit, width, height):
    loaded_images = []
    true_value = []
    for category in range(1, cat_num + 1):
        for image_id in range(1, limit[category - 1] + 1):
            path = source_path + str(category) + "/" + str(category) + "_" + str(image_id) + ".jpg"
            print(path)
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (width, height))
            image = image.flatten()
            image = image / 255
            loaded_images.append(image)

            probabilities = np.zeros(cat_num)
            probabilities[category - 1] = 1
            true_value.append(probabilities)
    training_image = np.array(loaded_images)
    true_values = np.array(true_value)
    return training_image, true_values

def model_a():
    inputs = imagenes
    targets = probabilidades
    kfold = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    # Lista para guardar el accuracy y el loss
    acc_per_fold = []
    loss_per_fold = []
    # Cross validation
    for train, test in kfold.split(inputs, targets):
        model = Sequential()
        # Capa entrada
        model.add(InputLayer(input_shape=(pixeles,)))
        model.add(Reshape(formaImagen))
        # Capas Ocultas
        # Capas convolucionales
        model.add(Conv2D(kernel_size=3, strides=2, filters=32, padding="same", activation="relu", name="capa_1"))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(kernel_size=3, filters=64, padding="same", activation="relu", name="capa_2"))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(kernel_size=3, filters=128, padding="same", activation="relu", name="capa_3"))
        model.add(MaxPool2D(pool_size=3, strides=2))
        model.add(Conv2D(kernel_size=3, filters=256, padding="same", activation="relu", name="capa_4"))
        model.add(MaxPool2D(pool_size=3, strides=2))
        '''
        model.add(Conv2D(kernel_size=3, filters=728, padding="same", activation="relu", name="capa_5"))
        model.add(MaxPool2D(pool_size=3, strides=2))
        '''
        # Aplanamiento
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        # Capa de salida
        model.add(Dense(numeroCategorias, activation="softmax"))
        # Traducir de keras a tensorflow
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # Entrenamiento
        model.fit(x=inputs[train], y=targets[train], epochs=20, batch_size=60)

        resultados = model.evaluate(x=inputs[test], y=targets[test])
        print("Accuracy=", resultados[1])
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {resultados[0]}; {model.metrics_names[1]} of {resultados[1] * 100}%')
        acc_per_fold.append(resultados[1] * 100)
        loss_per_fold.append(resultados[0])
        fold_no = fold_no + 1

        # Guardar modelo
        ruta = "models/modeloA.h5"
        model.save(ruta)
        # Informe de estructura de la red
        # model.summary()

    # == Resultados ==
    print('------------------------------------------------------------------------')
    print('Resultado por partición')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Carpeta {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Promedio de resultados de todas las particiones::')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')


def model_b():
    inputs = imagenes
    targets = probabilidades
    kfold = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    # Lista para guardar el accuracy y el loss
    acc_per_fold = []
    loss_per_fold = []

    for train, test in kfold.split(inputs, targets):
        model = Sequential()
        model.add(InputLayer(input_shape=(pixeles,)))
        model.add(Reshape(formaImagen))
        model.add(Conv2D(kernel_size=5, strides=2, filters=26, padding="same", activation="relu", name="capa_1"))
        model.add(MaxPool2D(pool_size=2, strides=2))

        model.add(Conv2D(kernel_size=18, strides=2, filters=36, padding="same", activation="relu", name="capa_2"))
        model.add(MaxPool2D(pool_size=2, strides=2))

        model.add(Conv2D(kernel_size=5, strides=2, filters=74, padding="same", activation="relu", name="capa_3"))
        model.add(MaxPool2D(pool_size=2, strides=2))

        # Aplanamiento
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))

        # Capa de salida
        model.add(Dense(numeroCategorias, activation="softmax"))

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # Entrenamiento
        model.fit(x=inputs[train], y=targets[train], epochs=30, batch_size=60)

        resultados = model.evaluate(x=inputs[test], y=targets[test])
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} de {resultados[0]}; {model.metrics_names[1]} de {resultados[1] * 100}%')
        acc_per_fold.append(resultados[1] * 100)
        loss_per_fold.append(resultados[0])
        fold_no = fold_no + 1

        # Guardar modelo
        ruta = "models/modeloB.h5"
        model.save(ruta)
        # Informe de estructura de la red
        # model.summary()

    # == Resultados ==
    print('------------------------------------------------------------------------')
    print('Resultado por partición')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Patición {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Promedio de resultados de todas las particiones::')
    print(f'> Accuracy: {np.mean(acc_per_fold)} perdida:(+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')


def model_c():
    inputs = imagenes
    targets = probabilidades
    kfold = KFold(n_splits=4, shuffle=True)
    fold_no = 1
    # Lista para guardar el accuracy y el loss
    acc_per_fold = []
    loss_per_fold = []

    for train, test in kfold.split(inputs, targets):
        model = Sequential()
        model.add(InputLayer(input_shape=(pixeles,)))
        model.add(Reshape(formaImagen))
        model.add(Conv2D(kernel_size=5, strides=2, filters=32, padding="same", activation="relu", name="capa_1"))
        model.add(MaxPool2D(pool_size=3, strides=2))
        model.add(Conv2D(kernel_size=5, strides=2, filters=64, padding="same", activation="relu", name="capa_2"))
        model.add(MaxPool2D(pool_size=3, strides=2))
        model.add(Conv2D(kernel_size=5, strides=2, filters=64, padding="same", activation="relu", name="capa_3"))
        model.add(MaxPool2D(pool_size=3, strides=2))
        # Aplanamiento
        # Aplanamiento
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        # Capa de salida
        model.add(Dense(numeroCategorias, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # Entrenamiento
        model.fit(x=inputs[train], y=targets[train], epochs=20, batch_size=60)

        # Evaluacion
        resultados = model.evaluate(x=inputs[test], y=targets[test])
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {resultados[0]}; {model.metrics_names[1]} of {resultados[1] * 100}%')
        acc_per_fold.append(resultados[1] * 100)
        loss_per_fold.append(resultados[0])
        fold_no = fold_no + 1
        # Guardar modelo
        ruta = "models/modeloC.h5"
        model.save(ruta)

    # == Resultados ==
    print('------------------------------------------------------------------------')
    print('Resultado por partición')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Carpeta {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Promedio de resultados de todas las particiones::')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')




#################################
ancho = 256
alto = 256
pixeles = ancho * alto
# Imagen RGB -->3
numeroCanales = 1
formaImagen = (ancho, alto, numeroCanales)
numeroCategorias = 5
cantidaDatosEntrenamiento = [180] * numeroCategorias
cantidaDatosPruebas = [24] * numeroCategorias

# Cargar las imágenes
imagenes, probabilidades = load_data("dataset/train/", numeroCategorias, cantidaDatosEntrenamiento, ancho, alto)
#model_c()
pred = Prediction("models/modeloC.h5", ancho, alto)
pred.metrics(imagenes, probabilidades)
