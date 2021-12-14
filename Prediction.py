from tensorflow.python.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import layer_utils, np_utils
import numpy as np
import cv2
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, log_loss, brier_score_loss

class Prediction():
    def __init__(self,ruta,ancho,alto):
        self.modelo=load_model(ruta)
        self.alto=alto
        self.ancho=ancho

    def predecir(self,imagen):
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen = cv2.resize(imagen, (self.ancho, self.alto))
        imagen = imagen.flatten()
        imagen = imagen / 255
        imagenesCargadas=[]
        imagenesCargadas.append(imagen)
        imagenesCargadasNPA=np.array(imagenesCargadas)
        predicciones=self.modelo.predict(x=imagenesCargadasNPA)
        print("Predicciones=",predicciones)
        clasesMayores=np.argmax(predicciones,axis=1)
        return clasesMayores[0]