import cv2
import mediapipe as mp
from deteccionGestos import identify_gesture
from letrasDinamicas import detectar_letra_dinamica

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None
buffer_movimiento = []
contador_visibilidad = 0
letra_actual = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)

    # Procesar el frame con MediaPipe para obtener landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    letra_estatica = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            handedness = results.multi_handedness[0].classification[0].label  # Detecta si es "Left" o "Right"
            letra_estatica = identify_gesture(hand_landmarks.landmark, handedness)

    # Detectar letras dinámicas (como la "Z")
    frame, buffer_movimiento, prev_x, prev_y, letra_dinamica = detectar_letra_dinamica(frame, buffer_movimiento, prev_x, prev_y)

    # Mostrar la letra detectada (prioridad a la dinámica)
    if letra_dinamica:
        letra_actual = letra_dinamica
        contador_visibilidad = 30
    elif letra_estatica:
        letra_actual = letra_estatica
        contador_visibilidad = 30

    if contador_visibilidad > 0:
        cv2.putText(frame, letra_actual, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        contador_visibilidad -= 1

    cv2.imshow("Detección de Gestos", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

