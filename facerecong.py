"""
Author: I
Date: today
"""
import os
from random import randint

import cv2 as cv
import numpy as np
from PIL import Image
from time import sleep
import face_recognition as fr

from multiprocessing import Process, Manager
from datetime import datetime as dt

from warnings import filterwarnings
filterwarnings('ignore')

from keras.models import load_model
from pyagender import PyAgender


# manejar variables compartidas entre procesos
manager = Manager()

faces_detected = manager.list()  # buffer para llenar con imagenes de caras
frames_buff = manager.list()  # buffer with all frames for process
frames_buff_out = manager.list()  # frames processed, ready for show
semaphore = manager.Semaphore()
face_processed = manager.list()  # faces processed for detect age

faces_folder = 'new_faces/'
faces_output = '.faces_detected_/'
faces_knows = [face for img in os.listdir(faces_folder) for face in fr.face_encodings(fr.load_image_file(faces_folder+img))]
know_people_count = len(faces_knows)

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


if not os.path.exists(os.path.join(os.curdir, faces_output)):
    os.mkdir(os.path.join(os.curdir, faces_output))


def how_are():
    """
    Busca detectar emociones en las caras guardadas
    """
    global faces_knows
    print('Runing emotion recognition')

    model = load_model('model_v6_23.hdf5')  # model trained for emotion detection
    file_processed = []
    while True:
        for face_file in os.listdir(faces_folder):

            if face_file in file_processed: continue

            img_face = cv.imread(faces_folder+face_file)
            # os.remove(faces_folder+face_file)

            if img_face is None : continue

            img_face = cv.cvtColor(img_face, cv.COLOR_BGR2GRAY)
            img_face = cv.resize(img_face, (48,48))
            img_face_emotion = np.reshape(img_face, [1, 48, 48, 1])
            predicted_class = np.argmax(model.predict(img_face_emotion))

            file_processed.append(face_file)

            cv.imwrite(f'process_faces/{emotions[predicted_class]}/{face_file}', img_face)
            print(f'Human {emotions[predicted_class]}')


def detect_gender_age():
    """
    detectar edad y genero
    """
    global face_processed

    agender = PyAgender()
    counter = 0

    def __gender(percent):
        return 'he' if percent < 0.5 else 'she'

    while True:
        if len(face_processed) > 0:
            img = face_processed.pop()
            faces = agender.detect_genders_ages(img)
            for i in range(len(faces)):
                print(f'{__gender(faces[i]["gender"])} is {faces[i]["age"]} years old')


def who_is():
    """
    Analiza cada cara en la cola faces_detected y las compara contra las caras que ya vio, si es nueva la guarda
    """
    global faces_detected, faces_knows, face_processed
    print('Runing face recognition')

    while True:
        if len(faces_detected) != 0:

            semaphore.acquire()
            new_face = faces_detected.pop()
            semaphore.release()

            faces_loc = fr.face_locations(new_face)
            new_face_encoding = fr.face_encodings(new_face)
            flag = False
            i = 0
            for face in new_face_encoding:
                results = fr.compare_faces(faces_knows, face, tolerance=0.5)
                print(results)
                if not True in results:
                    # Guardamos la cara nueva
                    x_1, y_1, x_2, y_2 = faces_loc[i]
                    face_detected = new_face[x_1:x_2, y_2:y_1][...,(2,1,0)]
                    face_detected = Image.fromarray(face_detected)
                    face_detected.save(faces_folder+f'face_{str(dt.now()).split(".")[0]}.jpg')

                    flag = True
                    print('New face add')

                    faces_knows = [face for img in os.listdir(faces_folder) for face in fr.face_encodings(fr.load_image_file(faces_folder+img))]
                    know_people_count = len(faces_knows)
                i += 1
            if flag:
                face_processed.append(new_face)


def save_fame_queue():
    global frames_buff, semaphore, faces_output
    img = frames_buff.pop()
    img = None
    img.save(faces_output + f'_{os.listdir(faces_output)}')
    return img


def __get_frame():
    """ Sacar un frame de buffer para procesamiento """
    global frames_buff, semaphore
    semaphore.acquire()
    if len(frames_buff) > 0:
        img = frames_buff.pop()
    else:
        img = None
    semaphore.release()
    return img


def count_humans():
    """
    Contar personas filmadas
    """
    global frames_buff, frames_buff_out
    print('Running human counter')

    haar_cascades = [
        'haarcascade_upperbody.xml',
        'haarcascade_fullbody.xml',
        'haarcascade_lowerbody.xml'
    ]
    body_upper = cv.CascadeClassifier(cv.data.haarcascades + haar_cascades[0])
    body_full = cv.CascadeClassifier(cv.data.haarcascades + haar_cascades[1])
    body_lower = cv.CascadeClassifier(cv.data.haarcascades + haar_cascades[2])

    counter = 0
    while True:
        img = __get_frame()
        if img is None: continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # rectangles = body_upper.detectMultiScale(gray, 1.1, 4)
        # rectangles = body_full.detectMultiScale(gray, 1.1, 4)
        rectangles = body_lower.detectMultiScale(gray, 1.1, 4)

#        rectangles = []
#        for (x, y, w, h) in bodys_full:
#            for (x_, y_, w_, h_) in bodys_upper:
#                # Evitar superposicion de rectangulos 10% cercanos
#                if (abs(x - x_)/img.shape[1]/100) < 10 and (abs(y - y_)/img.shape[1]/100) < 10:
#                    continue
#                rectangles.append((x_, y_, w_, h_))

#            for (x_, y_, w_, h_) in bodys_lower:
#                # Evitar superposicion de rectangulos 10% cercanos
#                if (abs(x - x_)/img.shape[1]/100) < 10 and (abs(y - y_)/img.shape[1]/100) < 10:
#                    continue
#                rectangles.append((x_, y_, w_, h_))
#            rectangles.append((x, y, w, h))

        for (x, y, w, h) in rectangles:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            frames_buff_out.append(img)
            counter += 1
            # print('Human detected: ', counter)


def detect_faces():
    """
    Detectar caras en un frame
    """
    global frames_buff_out, faces_detected
    print('Running detect faces')

    haar_cascades = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_profileface.xml',
        'haarcascade_frontalcatface_extended.xml',
        'haarcascade_frontalface_alt.xml',
        'haarcascade_frontalface_alt_tree.xml',
    ]

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + haar_cascades[0])
    face_profile = cv.CascadeClassifier(cv.data.haarcascades + haar_cascades[1])

    counter = 0
    while True:
        img = __get_frame()
        if img is None: continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces_front = face_cascade.detectMultiScale(gray, 1.1, 4)
        faces_profile = face_profile.detectMultiScale(gray, 1.1, 4)

        rectangles = []
        for (x, y, w, h) in faces_front:
            for (x_, y_, w_, h_) in faces_profile:
                # Evitar superposicion de rectangulos 10% cercanos
                if (abs(x - x_)/img.shape[1]/100) < 10 and (abs(y - y_)/img.shape[1]/100) < 10:
                    continue
                rectangles.append((x_, y_, w_, h_))
            rectangles.append((x, y, w, h))

        for (x, y, w, h) in rectangles:
            faces_detected.append(img) # [y:y+h, x:x+w,:]) # pasar solo la cara o toda la imagen?
            cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            frames_buff_out.append(img)


def show_frames():
    """
    Muestra los frames en una ventana
    """
    global frames_buff_out
    print('Running show frames')

    while True:
        if len(frames_buff_out) > 0:
            img = frames_buff_out.pop()
            cv.imshow('img', img)
            if (cv.waitKey(30) & 0xff) == 27:  # Key: Esc
                break


def film(cam_num=0, show=True):
    """
    Este metodo obtiene frames desde la camara, analiza cada uno buscando una cara de frente o perfil.
    Si encuentra encola la cara en una lista.
    """
    global faces_detected
    print('Recording')

    cam = cv.VideoCapture(cam_num)
    while True:

        ret, img = cam.read()
        if ret == False: continue

        frames_buff.append(img)

    cam.release()


if __name__=='__main__':
    print('-'*100, '\n'*3)
    print('Faces know: ', know_people_count)
    print('OpenCv version: ', cv.__version__)

    process = []
    functions = [film, detect_faces, show_frames, who_is, how_are, detect_gender_age]
    # functions = [film, detect_faces, count_humans, show_frames, who_is, how_are, detect_gender_age]

    # Lanzamos cada funcion en un proceso porque si
    i = 0
    for fun in functions:
        process.append(Process(target=fun))
        process[i].start()
        i += 1

    while True:
        sleep(2)
        if not process[0].is_alive():
            for p in process:
                p.terminate()
            break
