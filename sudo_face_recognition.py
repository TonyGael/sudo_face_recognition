# Paso 3: Reconocimiento facial y autenticación
# Este script utiliza las imágenes de referencia para autenticar el rostro capturado en tiempo real.

import cv2
import face_recognition
import numpy as np
import os
import subprocess

# directorio de imagenes de referencia
references_img_dir = 'references_images'

# cargamos las imagenes de referencia y aprendemos a reconocer las imagenes
know_face_encodings = []
know_face_names = []

for file_name in os.listdir(references_img_dir):
    if file_name.endswith('.png'):
        image_path = os.path.join(references_img_dir, file_name)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        know_face_encodings.append(encoding)
        know_face_names.append(file_name)

# inicializamos la captura de video
cap = cv2.VideoCapture(0)
cv2.namedWindow('Autenticacion facial')

while True:
    ret, frame = cap.read()
    if not ret:
        print('Error al acceder a la camara')
        break
    
    # convertimos la imagen de BGR a RGB
    rgb_frame = frame[:, :, ::-1]
    
    # encontramos los rostros y sus codificaciones en el frame actual
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(know_face_encodings, face_encoding)
        name = 'Desconocido'
        
        face_distances = face_recognition.face_distance(know_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = know_face_names[best_match_index]
            
            # si se reconoce el rostro ejecutamos un comando sudo
            print('Autenticando como:', name)
            subprocess.run(['sudo', 'apt', 'update'], check=True)
            break
    
    cv2.imshow('Autenticacion Facial', frame)
    
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # presiona Esc para salir
        break
    
cap.release()
cv2.destroyAllWindows()
