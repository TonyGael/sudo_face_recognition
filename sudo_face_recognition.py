# Paso 3: Reconocimiento facial y autenticación
# Este script utiliza las imágenes de referencia para autenticar el rostro capturado en tiempo real.
# pip install --upgrade setuptools

import cv2
import face_recognition
import numpy as np
import os
import subprocess
from PIL import Image

def process_image(image_path):
    # Abrimos la imagen con PIL
    image = Image.open(image_path)
    
    # Convertimos la imagen a RGB si no lo está
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convertimos la imagen a un array de numpy
    image_array = np.array(image)
    
    # Aseguramos que la imagen sea de 8 bits
    if image_array.dtype != np.uint8:
        image_array = (image_array / 256).astype(np.uint8)
    
    return image_array

# Directorio de imágenes de referencia
references_img_dir = 'references_images'

# Cargamos las imágenes de referencia y aprendemos a reconocer las caras
known_face_encodings = []
known_face_names = []

# Verificamos si el directorio de imágenes de referencia existe
if not os.path.exists(references_img_dir):
    print(f"El directorio {references_img_dir} no existe. Por favor, crea el directorio y añade las imágenes de referencia.")
    exit(1)

for file_name in os.listdir(references_img_dir):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(references_img_dir, file_name)
        try:
            # Procesamos la imagen
            image_array = process_image(image_path)
            
            # Obtenemos las codificaciones faciales
            encodings = face_recognition.face_encodings(image_array)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(file_name)
                print(f"Imagen {file_name} procesada con éxito.")
            else:
                print(f'Advertencia: No se encontraron rostros en la imagen {file_name}')
        except Exception as e:
            print(f"Error al procesar la imagen {file_name}: {str(e)}")

# Verificamos si se cargaron imágenes de referencia
if not known_face_encodings:
    print("No se pudieron cargar imágenes de referencia. Por favor, verifica las imágenes en el directorio.")
    exit(1)

print(f"Se cargaron {len(known_face_encodings)} imágenes de referencia.")

# Inicializamos la captura de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo acceder a la cámara. Por favor, verifica la conexión.")
    exit(1)

cv2.namedWindow('Autenticación facial')

while True:
    ret, frame = cap.read()
    if not ret:
        print('Error al acceder a la cámara')
        break
    
    # Convertimos la imagen de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Encontramos los rostros y sus codificaciones en el frame actual
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = 'Desconocido'
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
            # Si se reconoce el rostro, ejecutamos un comando sudo
            print('Autenticando como:', name)
            try:
                subprocess.run(['sudo', 'apt', 'update'], check=True)
                print("Comando sudo ejecutado con éxito.")
            except subprocess.CalledProcessError as e:
                print(f"Error al ejecutar el comando sudo: {e}")
            except PermissionError as e:
                print(f"Error de permisos al ejecutar sudo: {e}")
        
        # Dibujamos un rectángulo alrededor del rostro y mostramos el nombre
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    cv2.imshow('Autenticación Facial', frame)
    
    k = cv2.waitKey(1)
    if k % 256 == 27:  # Presiona Esc para salir
        print("Saliendo del programa...")
        break

cap.release()
cv2.destroyAllWindows()
print("Programa finalizado.")
