import cv2
import face_recognition
import numpy as np
import os
import subprocess

# Directorio de imágenes de referencia
reference_images_dir = "reference_images"

# Cargar imágenes de referencia y aprender a reconocerlas
known_face_encodings = []
known_face_names = []

for file_name in os.listdir(reference_images_dir):
    if file_name.endswith(".png"):
        image_path = os.path.join(reference_images_dir, file_name)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(file_name)

# Inicializar la captura de video
cap = cv2.VideoCapture(0)
cv2.namedWindow("Autenticación Facial")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara")
        break

    # Convertir la imagen de BGR a RGB
    rgb_frame = frame[:, :, ::-1]

    # Encontrar todas las caras y sus codificaciones en el frame actual
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Si se reconoce el rostro, ejecutar un comando con sudo
            print("Autenticado como:", name)
            subprocess.run(["sudo", "apt", "update"], check=True)
            break

    cv2.imshow("Autenticación Facial", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # Presiona Esc para salir
        break

cap.release()
cv2.destroyAllWindows()
