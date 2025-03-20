def identify_gesture(landmarks, lateralidad):
    
    # Pulgar
    base_pulgar = landmarks[1]
    proximal_pulgar = landmarks[2]
    medio_pulgar = landmarks[3]
    punta_pulgar = landmarks[4]
    # Índice
    base_indice = landmarks[5]
    proximal_indice = landmarks[6] 
    medio_indice = landmarks[7]
    punta_indice = landmarks[8]
    # Corazón
    base_corazon = landmarks[9]
    proximal_corazon = landmarks[10]
    medio_corazon = landmarks[11]
    punta_corazon = landmarks[12]
    # Anular
    base_anular = landmarks[13]
    proximal_anular = landmarks[14]
    medio_anular = landmarks[15]
    punta_anular = landmarks[16]
    # Meñique
    #jimmy esta roto
    base_menique = landmarks[17]
    proximal_menique = landmarks[18]
    medio_menique = landmarks[19]
    punta_menique = landmarks[20]
    
    # Verifica si los dedos están doblados hacia la palma
    si_indice_doblado = punta_indice.y > base_indice.y and abs(punta_indice.x - base_indice.x) < 0.05
    si_corazon_doblado = punta_corazon.y > base_corazon.y and abs(punta_corazon.x - base_corazon.x) < 0.05
    si_anular_doblado = punta_anular.y > base_anular.y and abs(punta_anular.x - base_anular.x) < 0.05
    si_menique_doblado = punta_menique.y > base_menique.y and abs(punta_menique.x - base_menique.x) < 0.05

    # Dedos levantados
    si_indice_levantado = punta_indice.y < medio_indice.y  # Índice levantado
    si_corazon_levantado = punta_corazon.y < medio_corazon.y  # Medio levantado
    si_anular_levantado = punta_anular.y < medio_anular.y # Anular levantado
    si_menique_levantado = punta_menique.y < medio_menique.y # Meñique levantado
    
    # Detección de la letra "A"
    pulgar_extendido_A = punta_pulgar.y < base_indice.y  
    if lateralidad == "Right":
        pulgar_al_lado_A = punta_pulgar.x < base_indice.x # Pulgar a la izquierda del índice
    elif lateralidad == "Left":
        pulgar_al_lado_A = punta_pulgar.x > base_indice.x # Pulgar a la derecha del índice  
    if pulgar_al_lado_A and si_indice_doblado and si_corazon_doblado and si_anular_doblado and si_menique_doblado and pulgar_extendido_A:
        return "A"
    
    
    # Deteccion de la letra "B"
    pulgar_escondido_indice_B = abs(punta_pulgar.y - base_indice.y) < 0.06 and abs(punta_pulgar.x - base_indice.x) < 0.04# El pulgar toca la base del anular
    if pulgar_escondido_indice_B and si_indice_levantado and  si_corazon_levantado and si_anular_levantado and si_menique_levantado:
        return "B"
    
    # Deteccion de la letra "C"
    estan_dedos_juntos_C = abs(punta_indice.y - punta_corazon.y) < 0.03 and abs(punta_corazon.y - punta_anular.y) < 0.03 and abs(punta_indice.x - punta_corazon.x) < 0.05 and abs(punta_corazon.x - punta_anular.x) < 0.3 # Dedos juntos
    profundidad_C = estan_dedos_juntos_C and abs(proximal_indice.z - proximal_corazon.z) > 0.02 and abs(proximal_corazon.z - proximal_anular.z) > 0.02
    pulgar_gancho_c = abs(punta_pulgar.x - punta_corazon.x) < 0.05
    if profundidad_C and pulgar_gancho_c and abs(punta_pulgar.y - punta_indice.y) > 0.11:
        return "C"

    # Detecciond de la letra "D"
    dedos_juntos_D = abs(punta_pulgar.x - punta_corazon.x) < 0.06 and abs(punta_pulgar.y - punta_corazon.y) < 0.06 and abs(punta_pulgar.x - punta_anular.x) < 0.06 and abs(punta_pulgar.y - punta_anular.y) < 0.06 and abs(punta_menique.y - punta_pulgar.y) < 0.06
    if si_indice_levantado and dedos_juntos_D:
        return "D"
    
    # Detección de la letra "E"
    garritas_E = abs(punta_indice.y - base_indice.y) < 0.06 and abs(punta_corazon.y - base_corazon.y) < 0.06 and abs(punta_anular.y - base_anular.y) < 0.06 and abs(punta_menique.y - base_menique.y) < 0.06
    if garritas_E and abs(punta_pulgar.x - base_indice.x) < 0.07 and abs(punta_pulgar.y - medio_pulgar.y) < 0.05:
        return "E"
   
    # Deteccion de la letra "F"
    indice_toca_pulgar = abs(punta_indice.x - punta_pulgar.x) < 0.05 and abs(punta_indice.y - punta_pulgar.y) < 0.05
    si_corazon_levantado = punta_corazon.y < medio_corazon.y 
    si_anular_levantado = punta_anular.y < medio_anular.y 
    si_menique_levantado = punta_menique.y < medio_menique.y  
    if indice_toca_pulgar and si_corazon_levantado and si_anular_levantado and si_menique_levantado:
        return "F"
    # Deteccion de la letra "G" Daniel
    

    # Deteccion de la letra "H"
    pistolita_H = abs(punta_anular.x - base_indice.x) < 0.06 and abs(base_corazon.y - punta_corazon.y) < 0.04 and abs(base_indice.y - punta_indice.y) < 0.04
    if pistolita_H and medio_pulgar.y < base_indice.y:
        return "H"

    # Detección de la letra "I"
    si_menique_levantado_I = proximal_menique.y < base_menique.y and medio_menique.y < proximal_menique.y and punta_menique.y < medio_menique.y
    pulgar_toca_dedo_I = abs(punta_pulgar.x - base_indice.x) < 0.07 and abs(punta_pulgar.y - base_indice.y) < 0.07
    dedos_juntos_I = abs(proximal_indice.x - proximal_corazon.x) < 0.07 and abs(proximal_indice.y - proximal_corazon.y) < 0.07 and abs(proximal_corazon.x - proximal_anular.x) < 0.07 and abs(proximal_corazon.y - proximal_anular.y) < 0.07
    if si_menique_levantado_I and pulgar_toca_dedo_I and dedos_juntos_I:
        return "I"
    
    # print("Meñique: ", si_menique_levantado_I)
    # print("Pulgar toca el dedo: ", pulgar_toca_dedo_I)
    # print("Dedos juntos: ", dedos_juntos_I)
    
    # Deteccion de la letra "J" later
    
    # Deteccion de la letra "K" later
    
    # Deteccion de la letra "L" Daniel 
    si_indice_levantado = punta_indice.y < base_indice.y          # Índice levantado
    if lateralidad == "Left":
        pulgar_horizontal = punta_pulgar.x > base_pulgar.x            # Pulgar extendido horizontalmente
    elif lateralidad == "Right":
        pulgar_horizontal = punta_pulgar.x < base_pulgar.x 


    if proximal_indice.y < base_indice.y and medio_indice.y < proximal_indice.y and punta_indice.y < medio_indice.y and pulgar_horizontal and si_corazon_doblado and si_anular_doblado and si_menique_doblado:
        return "L"
    
    # Detección de la letra "M" Daniel
    si_indice_apunta_abajo = punta_indice.y > medio_indice.y  # Índice apunta hacia abajo
    si_corazon_apunta_abajo = punta_corazon.y > medio_corazon.y  # Corazón apunta hacia abajo
    si_anular_apunta_abajo = punta_anular.y > medio_anular.y  # Anular apunta hacia abajo
    si_menique_doblado = punta_menique.y > medio_menique.y  # Meñique doblado hacia la palma

    # Verifica que el pulgar esté oculto
    pulgar_oculto = (
            punta_pulgar.y > base_indice.y and  # La punta del pulgar está debajo del índice
            punta_pulgar.y > base_corazon.y and  # La punta del pulgar está debajo del corazón
            abs(punta_pulgar.x - base_corazon.x) < 0.05  # El pulgar está alineado horizontalmente con los dedos extendidos
        )

    # Condicional estricta para detectar "M"
    if (si_indice_apunta_abajo and si_corazon_apunta_abajo and si_anular_apunta_abajo and si_menique_doblado and pulgar_oculto):
        return "M"
    
    # Deteccion de la letra "N" Daniel 
    if (si_indice_apunta_abajo and si_corazon_apunta_abajo and pulgar_oculto):
        return "N"
    # Deteccion de la letra "Ñ" later 
    
    # Detección de la letra "O" jimmysbroken 
    estan_dedos_juntos_O = abs(punta_indice.y - punta_corazon.y) < 0.03 and abs(punta_corazon.y - punta_anular.y) < 0.03 and abs(punta_indice.x - punta_corazon.x) < 0.05 and abs(punta_corazon.x - punta_anular.x) < 0.3 # Dedos juntos
    profundidad_O = estan_dedos_juntos_O and abs(proximal_indice.z - proximal_corazon.z) > 0.02 and abs(proximal_corazon.z - proximal_anular.z) > 0.02
    pulgar_gancho_O = abs(punta_pulgar.x - punta_indice.x) < 0.05
    if profundidad_O and pulgar_gancho_O and abs(punta_pulgar.y - punta_indice.y) < 0.07:
        return "O"
    
    # Deteccion de la letra "P" el_insano
    pulgar_toca_dedo_P = abs(punta_pulgar.y - proximal_corazon.y) < 0.06
    corazon_altura_indice_proximal_P = abs(punta_corazon.y - proximal_corazon.y) < 0.06
    if si_indice_levantado and pulgar_toca_dedo_P and corazon_altura_indice_proximal_P:
        return "P"
    
    # Deteccion de la letra "Q" el_insano
    
    indice_garrita_Q = punta_indice.y < base_indice.y and abs(punta_indice.y - proximal_indice.y) < 0.03
    print(indice_garrita_Q)
    if indice_garrita_Q and si_corazon_doblado and si_anular_doblado and si_menique_doblado and pulgar_horizontal:
        return "Q"
    
    # Deteccion de la letra "R" el_insano
    pulgar_punta_toca_anular_punta_R = abs(punta_pulgar.y - punta_anular.y) < 0.07 and abs(punta_pulgar.x - punta_anular.x) < 0.07
    indice_junto_corazon_R = abs(punta_indice.x - punta_corazon.x) < 0.04
    if pulgar_punta_toca_anular_punta_R and si_indice_levantado and si_corazon_levantado and si_anular_doblado and si_menique_doblado and indice_junto_corazon_R:
        return "R"
    
    # Deteccion de la letra "S"jimmysbroken
    if lateralidad == "Right":
        pulgar_en_medio_S = punta_pulgar.x > proximal_indice.x and abs(punta_pulgar.x - proximal_corazon.x) < 0.03 and punta_pulgar.y >= proximal_corazon.y
    if lateralidad == "Left":
        pulgar_en_medio_S = punta_pulgar.x < proximal_indice.x and abs(punta_pulgar.x - proximal_corazon.x) < 0.03 and punta_pulgar.y >= proximal_corazon.y
        
    if pulgar_en_medio_S and si_indice_doblado and si_corazon_doblado and si_anular_doblado and si_menique_doblado:
        return "S"
    
    # Deteccion de la letra "T" jimmysbroken
    if lateralidad == "Right":
        tengo_tu_nariz_t = punta_pulgar.x > proximal_indice.x and punta_pulgar.x < proximal_corazon.x and punta_pulgar.y < proximal_corazon.y
    if lateralidad == "Left":
        tengo_tu_nariz_t = punta_pulgar.x < proximal_indice.x and punta_pulgar.x > proximal_corazon.x and punta_pulgar.y < proximal_corazon.y
        
    if tengo_tu_nariz_t and si_indice_doblado and si_corazon_doblado and si_anular_doblado and si_menique_doblado and pulgar_extendido_A:
        return "T"
    # Detección de la letra "U"
    si_indice_levantado = punta_indice.y < base_indice.y  # Índice levantado
    si_corazon_levantado = punta_corazon.y < base_corazon.y  # Medio levantado
    estan_dedos_juntos = abs(punta_indice.x - punta_corazon.x) < 0.05  # Índice y medio juntos
    if si_indice_levantado and si_corazon_levantado and estan_dedos_juntos and si_anular_doblado and si_menique_doblado:
        return "U"
    
    # Deteccion de la letra "V" Daniel 
    si_indice_levantado = punta_indice.y < base_indice.y            
    si_corazon_levantado = punta_corazon.y < base_corazon.y          
    si_anular_doblado = punta_anular.y > base_anular.y              
    si_menique_doblado = punta_menique.y > base_menique.y

    if si_indice_levantado and si_corazon_levantado and si_anular_doblado and si_menique_doblado:
        return "V"
    
    # Deteccion de la letra "W" el_insano
    pulgar_punta_toca_menique_medio_W = abs(punta_pulgar.x - medio_menique.x) < 0.06 and abs(punta_pulgar.y - medio_menique.y) < 0.07
    if pulgar_punta_toca_menique_medio_W and si_menique_doblado and si_anular_levantado and si_corazon_levantado and si_indice_levantado:
        return "W"
    
    # Deteccion de la letra "X" el_insano
    si_indice_doblado_X = abs(punta_indice.y - proximal_indice.y) > 0.06  # El índice está doblado hacia la palma
    puño_cerrado_X = abs(punta_corazon.y - base_corazon.y) < 0.06 and abs(punta_anular.y - base_anular.y) < 0.06 and abs(punta_menique.y - base_menique.y) < 0.06
    if si_indice_doblado_X and puño_cerrado_X:
        return "X"
    
    # Deteccion de la letra "Y" jimmysbroken
    if lateralidad == "Right":
        drinks = proximal_menique.x < base_menique.x and medio_menique.x < proximal_menique.x and punta_menique.x < medio_menique.x and medio_anular.x > proximal_menique.x and medio_corazon.x > proximal_menique.x and abs(proximal_corazon.x - proximal_anular.x) < 0.02
    if lateralidad == "Left":
        drinks = proximal_menique.x > base_menique.x and medio_menique.x < proximal_menique.x and punta_menique.x > medio_menique.x and medio_anular.x < proximal_menique.x and medio_corazon.x < proximal_menique.x and abs(proximal_corazon.x - proximal_anular.x) < 0.02
    if drinks and punta_pulgar.y < medio_pulgar.y and abs(punta_pulgar.x - medio_pulgar.x) < 0.03:
        return "Y"
    
    