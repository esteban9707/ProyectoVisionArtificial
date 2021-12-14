import threading

import numpy as np
import cv2

import GUI
import imageConverter
from Image import Image
from Client import Client
import uuid
import json
import requests
from GUI import inicialize
import main
import testCNN

nameWindow ="Calculadora Canny"
def nothing(x):
    pass

def construirVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min",nameWindow,0,255,nothing)
    cv2.createTrackbar("max",nameWindow,1,100,nothing)
    cv2.createTrackbar("kernel",nameWindow,0,255,nothing)
    cv2.createTrackbar("areaMin",nameWindow,500,10000,nothing)


def calcularAreas(objetos):
    areas=[]
    for objetoActual in objetos:
        areas.append(cv2.contourArea(objetoActual))
    return areas



def detectarForma(imagen,imagen_countours,img_name, predict=False):
    # Reducir dimensiones de la imagen de 3D a 2D
    # Conversión a escala de grises
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Imagen Gris", imagenGris)
    # Hallar los bordes
    # Utilizando la derivada (Canny)
    min = cv2.getTrackbarPos("min", nameWindow)
    max = cv2.getTrackbarPos("max", nameWindow)
    bordes = cv2.Canny(imagenGris, min, max)
    #cv2.imshow("Bordes", bordes)
    # Operaciones morfológicas
    tamañoKernel = cv2.getTrackbarPos("kernel", nameWindow)
    kernel = np.ones((tamañoKernel, tamañoKernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    #cv2.imshow("Bordes reforzados ", bordes)
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = calcularAreas(figuras)
    areaMinima = cv2.getTrackbarPos("areaMin", nameWindow)
    i = 0
    for figuraActual in figuras:
        if (areas[i] >= areaMinima):
            vertices = cv2.approxPolyDP(figuraActual, 0.05 * cv2.arcLength(figuraActual, True), True)
            ##print(vertices[0][0])
            if (len(vertices) == 3):
                pass
                # mensaje="Triangulo"
                # cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.drawContours(imagen, [figuraActual], 0, (0, 0, 255), 2)
            elif (len(vertices) == 4):
                mensaje = "Billete de fuck detectado"
                #print("cuadrilatero")
                cv2.putText(imagen_countours, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.drawContours(imagen_countours, [figuraActual], 0, (0, 0, 255), 2)

                x, y, w, h = cv2.boundingRect(vertices)
                new_img = imagen[y:y + h, x:x + w]
                if(predict):
                    cv2.imwrite(img_name, new_img)
                    return new_img
            elif (len(vertices) == 5):
                pass
                # mensaje="Pentagono"
                # cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.drawContours(imagen, [figuraActual], 0, (0, 0, 255), 2)
    return imagen


camara = cv2.VideoCapture(1)
construirVentana()
img_counter = 1
billete = "0"
cont = 0
imgencodes=[]
imagesrequest=[]
m3 = None
m2 = None
m1 = None
id_counter = 0
th = threading.Thread(target=inicialize)  # initialise the thread
th.setDaemon(True)
th.start()  # start the thread

def addModels():
    if (GUI.m1.get() == 1):
        models.append(1)
    if (GUI.m2.get() == 1):
        models.append(2)
    if (GUI.m3.get() == 1):
        models.append(3)

def sendRequest():
    global models, imgencodes, imagesrequest, id_counter
    models = []
    addModels()
    for imgs64 in imgencodes:
        idImage = id_counter
        img = {
            "id": idImage,
            "content": imgs64
        }
        imagesrequest.append(img)
        id_counter += 1
    id_counter = 1
    idClient = str(uuid.uuid4())
    clientRequest = {
        "id_client": idClient,
        "models": models,
        "images": imagesrequest
    }
    resp = requests.post('http://localhost:5000/predict', json=clientRequest)
    GUI.global_response = resp.content
    GUI.setText(resp.content)
    imgencodes = []
    imagesrequest = []


while True:
    _, imagen = camara.read()
    _, imagen_countours = camara.read()

    imagen = detectarForma(imagen, imagen_countours, '')



   # cv2.putText(imagen, f'El valor del  es: {billete}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Imagen Camara", imagen)
    cv2.imshow("Bordes", imagen_countours)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if k % 256 == 99:
        img_name = "imagen_{}.jpg".format(img_counter)
        imagen = detectarForma(imagen,imagen_countours,img_name, True)
        cv2.imwrite(img_name, imagen)
        imgencode = imageConverter.img_to_base64(img_name)
        imgencodes.append(imgencode)
        img_counter += 1

    if k % 256 == 101:
        sendRequest()

camara.release()
cv2.destroyAllWindows()
