import pandas as pd
import os
import cv2
from tqdm import tqdm

# Ruta del archivo de video
video_path = '/Users/aitanarilla/Desktop/Modelo_caras/Files/E3_camara1_2000-01-01_00.04.02.mp4'

# Abrir h5 y obtener coor x e y
h5_file = '/Users/aitanarilla/Desktop/Modelo_caras/Files/E3_camara1_2000-01-01_00.04.02DLC_resnet50_CIA_ratsNov19shuffle1_1030000.h5'
line_number = 18660
body_parts = {'Ojo_izquierdo': (0, 0, 255),  # Rojo
              'Ojo_derecho': (0, 255, 0),  # Verde
              'Nariz': (255, 0, 0)}       # Azul

# Read the .h5 file and load it into a Pandas DataFrame
try:
    df = pd.read_hdf(h5_file)
    
    # Access the coordinates for the th frame
    coordinates = {}
    for part, color in body_parts.items():
        x = df.loc[line_number, ('DLC_resnet50_CIA_ratsNov19shuffle1_1030000', part, 'x')]
        y = df.loc[line_number, ('DLC_resnet50_CIA_ratsNov19shuffle1_1030000', part, 'y')]
        coordinates[part] = (x, y, color)

    print(f"Coordinates for the {line_number}th frame:")
    for part, (x, y, _) in coordinates.items():
        print(f"{part}: x={x}, y={y}")

except FileNotFoundError:
    print(f"The file {h5_file} was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while trying to read the .h5 file: {str(e)}")
    exit()

# Abre el video
cap = cv2.VideoCapture(video_path)

# Verifica si el video se abrió correctamente
if not cap.isOpened():
    print("Error al abrir el video")
    cap.release()
    exit()

#Verificar el número de frames del video (opcional)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total de frames en el video: {total_frames}")

# Itera hasta llegar al frame deseado ()
for i in tqdm(range(total_frames)):  
    ret, frame = cap.read()
    if i == (line_number):  
        for part, (x, y, color) in coordinates.items():
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            cv2.imwrite(f'Frame_{line_number}.png', frame)

    if not ret:
        break

# Libera los recursos y cierra la ventana de visualización
cap.release()
cv2.destroyAllWindows()
