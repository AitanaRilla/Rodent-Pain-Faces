import pandas as pd
import os
import cv2
from tqdm import tqdm

def verificar_crear_directorio(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)
        print(f"Directorio '{ruta}' creado exitosamente.")  

def procesar_video_y_h5(root_path, nivel=0, max_nivel=None):
    for elemento in os.listdir(root_path):
        elemento_path = os.path.join(root_path, elemento)
        if os.path.isdir(elemento_path):
            if max_nivel is None or nivel < max_nivel:
                # Checks if the folder contains files with .mp4 extension
                archivos_mp4 = [f for f in os.listdir(elemento_path) if f.endswith(".mp4")]
                if archivos_mp4:
                    procesar_carpeta(elemento_path)
                # Calls the function recursively to scan subdirectories
                procesar_video_y_h5(elemento_path, nivel=nivel+1, max_nivel=max_nivel)

def procesar_carpeta(carpeta_path):
    if not os.path.isdir(carpeta_path):  # Check if it's a directory
        return

    # Get list of files in the folder
    archivos = os.listdir(carpeta_path)

    for nombre_fichero_mp4 in [f for f in archivos if f.endswith(".mp4")]:
        nombre_fichero = nombre_fichero_mp4[:-4]  # Removes the .mp4 extension
        video_path = os.path.join(carpeta_path, nombre_fichero_mp4)

        # Search for corresponding .h5 file
        h5_files = [f for f in archivos if f.startswith(nombre_fichero) and f.endswith('.h5')]
        if not h5_files:
            print(f"No se encontró ningún archivo h5 correspondiente para {nombre_fichero}.")
            continue
        h5_file = f"{carpeta_path}/{h5_files[0]}"

        # Path to save the frames
        directorio_destino = os.path.join(carpeta_path, nombre_fichero)
        verificar_crear_directorio(directorio_destino)

        # Read the .h5 file and load it into a Pandas DataFrame
        try:
            df = pd.read_hdf(h5_file)
        except FileNotFoundError:
            print(f"The file {h5_file} was not found.")
            return
        except Exception as e:
            print(f"An error occurred while trying to read the .h5 file: {str(e)}")
            return

        # Obtain all coordinates for each frame
        frame_coordinates = {}
        for frame_number in tqdm(range(len(df))):  
            coordinates = {}
            if frame_number % 10 == 0:
                for part, color in body_parts.items():
                    x = df.loc[frame_number, ('DLC_resnet50_CIA_ratsNov19shuffle1_1030000', part, 'x')]
                    y = df.loc[frame_number, ('DLC_resnet50_CIA_ratsNov19shuffle1_1030000', part, 'y')]
                    likelihood = df.loc[frame_number, ('DLC_resnet50_CIA_ratsNov19shuffle1_1030000', part, 'likelihood')]
                    coordinates[part] = (x, y, likelihood, color)
                    frame_coordinates[frame_number] = coordinates

        # Open the video
        cap = cv2.VideoCapture(video_path)
        cap2 = cv2.VideoCapture(video_path)

        # Check if the video was opened correctly
        if not cap.isOpened():
            print("Error al abrir el video")
            cap.release()
            return

        # Check the number of frames of the video (optional)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total de frames en el video: {total_frames}")

        # Process each frame
        for i in tqdm(range(total_frames)):  
            ret, frame = cap.read()
            ret2, frame2 = cap2.read()
            if not ret:
                break
            if i % 10 == 0:
                if i in frame_coordinates:  
                    for part, ((r, g, b), square_size) in body_parts.items():
                        x, y, likelihood, _ = frame_coordinates[i][part]
                        half_size = square_size // 2
                        top_left = (int(x - half_size), int(y - half_size))
                        bottom_right = (int(x + half_size), int(y + half_size))
                        
                        cv2.rectangle(frame2, top_left, bottom_right, (r, g, b), 2)

                        if likelihood > 0.7: 
                            try:
                                print(f"\033[92mSe procesa frame {i}\033[0m")
                                cropped_frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                                cv2.imwrite(f'{directorio_destino}/Frame_{i}_{part}.png', cropped_frame)
                            except Exception as e:
                                print(f"Error al procesar el frame {i}: {str(e)}")
                        else:
                            print(f"\033[93mNo se procesa frame {i}\033[0m")
                            print(f"x={x} y={y} likelihood={likelihood}")

                        if likelihood > 0.7:            
                            cv2.imwrite(f'{directorio_destino}/Frame_{i}.png', frame2)

        # Frees the resources and closes the view window
        cap.release()
        cap2.release()
        cv2.destroyAllWindows()

body_parts = {
    'Ojo_izquierdo': ((0, 0, 255), 50),  # 'Left eye', Red, size 50
    'Ojo_derecho': ((0, 255, 0), 60),   # 'Right eye', Green, size 50
}

root_path = "/Users/aitanarilla/Desktop/Modelo_caras/Files"
procesar_video_y_h5(root_path, max_nivel=3)  # Change max_level according to your needs
