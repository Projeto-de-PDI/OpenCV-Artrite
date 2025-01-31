import cv2
import mediapipe as mp
import numpy as np
import joblib

# Carregar modelo SaLVO
model = joblib.load('C:/Users/jmarques/Desktop/UFMA/PDI/OpenCV-Artrite/EXERCICIO1/Modelo_Exercicio1.pkl')

# Configurar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture('C:/Users/jmarques/Desktop/UFMA/PDI/OpenCV-Artrite/EXERCICIO1/DATA_SAMPLE\DATA6.mp4')  # Pode ser substituído por 0 para webcam
gestures_sequence = []

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Processamento de imagem
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Landmarks das mãos
                landmarks = [landmark for landmark in hand_landmarks.landmark]
                flattened = [coord for landmark in landmarks for coord in (landmark.x, landmark.y, landmark.z)]

                # Predição com o modelo.
                prediction = model.predict([flattened])[0]
                gesture = "aberta" if prediction == 0 else "fechada"

                # Adicionar gesto à sequência, mas apenas se for diferente do último
                if not gestures_sequence or gestures_sequence[-1] != gesture:
                    gestures_sequence.append(gesture)
                
                # Manter a lista com no máximo 3 elementos
                if len(gestures_sequence) > 3:
                    gestures_sequence.pop(0)

                # Verificar se a sequência corresponde ao "Exercício 1"
                if gestures_sequence == ["aberta", "fechada", "aberta"]:
                    cv2.putText(image, "Exercicio 1", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # gestures_sequence = []

                # Desenho e Texto
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(image, gesture, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Gesture Recognition', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("Sequência de gestos detectada:", gestures_sequence)
