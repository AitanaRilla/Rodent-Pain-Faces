import pandas as pd
import os
import cv2
from tqdm import tqdm

#Path of videos, change if moving
video_path = '/Users/aitanarilla/Desktop/Modelo_caras/Files/E3_camara1_2000-01-01_00.04.02.mp4'

#Open .h5 file and get coordinates x and y from body parts
h5_file = '/Users/aitanarilla/Desktop/Modelo_caras/Files/E3_camara1_2000-01-01_00.04.02DLC_resnet50_CIA_ratsNov19shuffle1_1030000.h5'
line_number = 380
body_parts = {
    'Ojo_izquierdo': ((0, 0, 255), 50), #Red, square size 50
    'Ojo_derecho': ((0, 255, 0), 50), #Green, square size 50
    'Nariz': ((255, 0, 0), 60) #Blue, square size 60
}

#Read the .h5 file and load it into a DataFrame
try:
    df = pd.read_hdf(h5_file)
    
    #Get the coordinates for the []th frame
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

#Check if the video opened correctly
if not cap.isOpened():
    print("Error opening the video")
    cap.release()
    exit()

#Check the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames on the video: {total_frames}")

for i in tqdm(range(total_frames)):  
    ret, frame = cap.read()
    if i == (line_number):  
        for part, ((r, g, b), square_size) in body_parts.items():
            x, y, _ = coordinates[part]
            half_size = square_size // 2
            top_left = (int(x - half_size), int(y - half_size))
            bottom_right = (int(x + half_size), int(y + half_size))
            cv2.rectangle(frame, top_left, bottom_right, (r, g, b), 2)
            cv2.imwrite(f'Frame_{line_number}.png', frame)

    if not ret:
        break

#Frees resources and closes the display window
cap.release()
cv2.destroyAllWindows()
