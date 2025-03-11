import cv2
import mediapipe as mp
import time
from deteccionGestos import identify_gesture  # Importamos la función

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Inicializa la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
          for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lateralidad = hand_info.classification[0].label  # "Right" o "Left"
            confidence = hand_info.classification[0].score  # Confianza de la clasificación
            # Llama a la función de identificación de gestos
            gesture = identify_gesture(hand_landmarks.landmark, lateralidad)
            
             # Mostrar si es mano derecha o izquierda en la parte superior derecha
            text = f'{lateralidad} ({confidence:.2f})'
            cv2.putText(frame, text, (frame.shape[1] - 200, 50),  # Posición en la esquina superior derecha
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Mostrar texto solo si se detecta un gesto
            if gesture is not None:
                cv2.putText(frame, f'Gesto: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()