import cv2
import os

# directorio donde se guardarán las imágenes de referencia
reference_images_dir = 'reference_images'

# crear el driectorio si no existe
if not os.path.exist(reference_images_dir):
    os.makedirs(reference_images_dir)

cap = cv2.VideoCapture(0)
cv2.namedWindow('Captura de Referencia:')

img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print('Error al acceder a la camara')
        break
    
    cv2.imshow('Captura de Referencia', frame)
    
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # presiona Esc para salir
        break
    elif k % 256 == 32:
        # Presiona espacio para capturar una imagen
        img_name = 'f{reference_umages_dir}/ref_image_{img_counter}.png'
        cv2.imwrite(img_name, frame)
        print(f'{img_name} guardada!')
        img_counter += 1

cap.release()
cv2.destroyAllWindows()

