import pandas as pd
import os
import cv2

#Path of videos, change if moving
video_path = '/Users/aitanarilla/Desktop/Modelo_caras/Files/E3_camara3_2022-04-22_08.43.11.mp4'

#Open .h5 file and get coordinates x and y from body parts
h5_file = '/Users/aitanarilla/Desktop/Modelo_caras/Files/E3_camara3_2022-04-22_08.43.11DLC_resnet50_CIA_ratsNov19shuffle1_1030000.h5'
line_number = 264
body_parts = {'Ojo_izquierdo': (0, 0, 255),  # Rojo
              'Ojo_derecho': (0, 255, 0),  # Verde
              'Nariz': (255, 0, 0)}       # Azul

#Read the .h5 file and load it into a Pandas DataFrame
try:
    df = pd.read_hdf(h5_file)
    
    #Access the coordinates for the xth frame
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

#Open the video
cap = cv2.VideoCapture(video_path)

#Check if the video opened correctly and frees resources and closes the display window
if not cap.isOpened():
    print("Error al abrir el video")
    cap.release()
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total de frames en el video: {total_frames}")