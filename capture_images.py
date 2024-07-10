import cv2
import os
import numpy as np
from PIL import Image

# Directorio donde guardar las imágenes de referencia
reference_images_dir = 'references_images'

# Creamos el directorio si no existe
if not os.path.exists(reference_images_dir):
    os.makedirs(reference_images_dir)

def process_and_save_image(frame, filename):
    # Convertimos el frame de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convertimos a imagen PIL
    pil_image = Image.fromarray(rgb_frame)
    
    # Aseguramos que la imagen esté en modo RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convertimos de vuelta a un array numpy
    image_array = np.array(pil_image)
    
    # Aseguramos que la imagen sea de 8 bits
    if image_array.dtype != np.uint8:
        image_array = (image_array / 256).astype(np.uint8)
    
    # Guardamos la imagen
    cv2.imwrite(filename, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    print(f'{filename} guardada!')

cap = cv2.VideoCapture(0)
cv2.namedWindow('Captura de referencia')

img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print('Error al acceder a la cámara')
        break
    
    cv2.imshow('Captura de referencia', frame)
    
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # Presiona Esc para salir
        print("Cerrando...")
        break
    elif k % 256 == 32:
        # Presiona espacio para capturar una imagen
        img_name = f'{reference_images_dir}/ref_image_{img_counter}.png'
        process_and_save_image(frame, img_name)
        img_counter += 1

cap.release()
cv2.destroyAllWindows()

print(f"Se capturaron {img_counter} imágenes de referencia.")
print("Asegúrate de que las imágenes capturadas muestren claramente los rostros.")
print("Programa finalizado.")
