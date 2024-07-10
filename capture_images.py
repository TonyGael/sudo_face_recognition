# Paso 1: Captura y almacenamiento de imágenes de referencia

import cv2
import os

# directorio donde guardar las imagenes de referencia
reference_images_dir = 'references_images'

# creamos el repositorio si no existe
if not os.path.exists(reference_images_dir):
    os.makedirs(reference_images_dir)


cap = cv2.VideoCapture(0)
cv2.namedWindow('Captura de referencia')

img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print('Error al acceder a la camara')
        break
    
    cv2.imshow('Captura de referencia: ', frame)
    
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # presiona Esc para salir
        break
    elif k % 256 == 32:
        # presiona espacio para capturar una imagen
        img_name = f'{reference_images_dir}/ref_image_{img_counter}.png'
        cv2.imwrite(img_name, frame)
        print(f'{img_name} guardada!')
        img_counter += 1

cap.release()
cv2.destroyAllWindows()
