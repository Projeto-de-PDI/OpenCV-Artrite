import cv2
import mediapipe as mp
import numpy as np
import math
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

cap = cv2.VideoCapture('DATA_SAMPLE/DATA6.mp4')
data = []

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.3) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Processamento da imagem
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Captura e desenho de landmarks
        if results.multi_hand_landmarks:
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):

                ## Definir rótulo para mão aberta/fechada
                hand_label = results.multi_handedness[num].classification[0].label

                wrist = hand_landmarks.landmark[0]
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]

                thumb_to_wrist = calculate_distance(thumb_tip, wrist)
                index_to_wrist = calculate_distance(index_tip, wrist)
                middle_to_wrist = calculate_distance(middle_tip, wrist)

                if thumb_to_wrist > 0.35 and index_to_wrist > 0.35 and middle_to_wrist > 0.35:
                    gesture = "aberta"
                else:
                    gesture = "fechada"

                cv2.putText(image, f"{hand_label}: {gesture}", (10, 50 + num * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Coleta das coordenadas e rótulo (mão aberta ou fechada)
                landmarks = [landmark for landmark in hand_landmarks.landmark]
                flattened = [coord for landmark in landmarks for coord in (landmark.x, landmark.y, landmark.z)]

                # Adicionar o rótulo junto com as coordenadas
                data.append(flattened + [gesture])  # Adiciona o rótulo ao final das coordenadas

        cv2.imshow('Frame', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Agora você pode salvar os dados com os rótulos em um arquivo CSV
df = pd.DataFrame(data, columns=[f'landmark_{i}_x' for i in range(21)] + [f'landmark_{i}_y' for i in range(21)] + [f'landmark_{i}_z' for i in range(21)] + ['gesture'])
df.to_csv('Exerc1.csv', index=False)
