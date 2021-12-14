import cv2
from  Prediction import Prediction
import time

def probarModelo(imagen):
    print("holla")
    clases = ["Mil","Dos mil", "5 mil", "10 mil", "20 mil"]
    ancho = 128
    alto = 128
    miModeloCNN = Prediction("models/modeloA.h5", ancho, alto)
    #imagen = cv2.imread("dataset/train/4/4_120.jpg")
    imagen = cv2.imread(imagen)
    start_time = time.time()
    claseResultado = miModeloCNN.predecir(imagen)
    end_time = time.time() - start_time
    print("--- tiempo en predecir: %s segundos ---" % end_time)
    print("El valor del billete cargado es " + clases[claseResultado] )
    return clases[claseResultado]

