import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def detectar_letra_dinamica(frame, buffer_movimiento, prev_x, prev_y):
    """
    Detecta letras dinámicas basadas en el movimiento de la mano.
    :param frame: Frame de la cámara.
    :param buffer_movimiento: Lista con puntos previos para analizar el movimiento.
    :param prev_x: Última coordenada X detectada.
    :param prev_y: Última coordenada Y detectada.
    :return: Frame procesado, buffer_movimiento actualizado, prev_x, prev_y, letra_detectada
    """
    letra_detectada = None
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            if prev_x is not None and prev_y is not None:
                dx, dy = x - prev_x, y - prev_y
                buffer_movimiento.append((dx, dy))
                cv2.line(frame, (prev_x, prev_y), (x, y), (255, 0, 0), 2)
                
                letra_detectada = detectar_letra_z(buffer_movimiento)
                
                if len(buffer_movimiento) > 20:
                    buffer_movimiento.pop(0)
            
            prev_x, prev_y = x, y
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return frame, buffer_movimiento, prev_x, prev_y, letra_detectada

def detectar_letra_z(movimientos):
    if len(movimientos) < 20:
        return None
    
    mov_x = [m[0] for m in movimientos]
    mov_y = [m[1] for m in movimientos]
    
    if (mov_x[0] > 3 and abs(mov_y[0]) < 5 and  # Horizontal derecha
        mov_x[7] < -3 and mov_y[7] > 5 and      # Diagonal izquierda-abajo
        mov_x[14] > 3 and abs(mov_y[14]) < 5):  # Horizontal derecha
        return "Z"
    
    return None