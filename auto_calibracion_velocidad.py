"""
auto_calibracion_velocidad.py

Usa el modelo Soccana_Keypoint para detectar automáticamente
los puntos clave de la cancha (esquinas del área grande),
calcula la homografía y con eso la velocidad del balón en m/s y km/h.
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


# ============ CONFIGURACIÓN ============

# Ruta local al modelo Soccana_Keypoint (descargalo desde HuggingFace)
# https://huggingface.co/Adit-jain/Soccana_Keypoint
MODEL_PATH = "soccana_keypoint.pt"

# Medidas reales del área grande (en metros)
# Área grande estándar FIFA: 40.32 m de ancho x 16.5 m de profundidad
ANCHO_AREA_M = 40.32
ALTO_AREA_M = 16.5

# Coordenadas reales de las 4 esquinas del área (en metros)
# Orden: sup-izq, sup-der, inf-der, inf-izq
FIELD_POINTS_M = np.array([
    [0.0,            0.0],
    [ANCHO_AREA_M,   0.0],
    [ANCHO_AREA_M,   ALTO_AREA_M],
    [0.0,            ALTO_AREA_M],
], dtype=np.float32)

# Índices de los keypoints del modelo para el área grande
# Según Soccana_Keypoint (29 puntos):
#   Área izquierda: 1, 2, 3, 4
#   Área derecha:  17, 18, 19, 20
KEYPOINTS_AREA_IZQ = [1, 2, 3, 4]
KEYPOINTS_AREA_DER = [17, 18, 19, 20]

# Umbral mínimo de visibilidad de un keypoint
VISIBILIDAD_MIN = 0.5


# ============ FUNCIONES ============

def cargar_modelo():
    """Carga el modelo Soccana_Keypoint."""
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {MODEL_PATH}\n"
            f"Descargalo desde: https://huggingface.co/Adit-jain/Soccana_Keypoint"
        )
    return YOLO(MODEL_PATH)


def detectar_keypoints_cancha(modelo, frame):
    """
    Detecta los keypoints del área grande en el frame.
    Devuelve un array (4, 2) con los píxeles o None si no se pudo.
    """
    resultados = modelo(frame, verbose=False)
    if not resultados:
        return None

    resultado = resultados[0]
    if resultado.keypoints is None:
        return None

    kpts = resultado.keypoints.data.cpu().numpy()  # (N, 29, 3)
    if len(kpts.shape) == 3:
        kpts = kpts[0]  # (29, 3)

    # Intentar primero el área izquierda, si falla usar la derecha
    for indices in (KEYPOINTS_AREA_IZQ, KEYPOINTS_AREA_DER):
        puntos = []
        ok = True
        for idx in indices:
            x, y, vis = kpts[idx]
            if vis < VISIBILIDAD_MIN:
                ok = False
                break
            puntos.append([x, y])
        if ok:
            return np.array(puntos, dtype=np.float32)

    return None


def calcular_homografia(puntos_px, puntos_m):
    """Calcula la homografía con RANSAC."""
    H, _ = cv2.findHomography(puntos_px, puntos_m, cv2.RANSAC, 5.0)
    if H is None or not np.isfinite(H).all():
        return None
    return H


def pixel_a_metros(punto_px, H):
    """Convierte un punto (x, y) en píxeles a metros."""
    p = np.array([punto_px[0], punto_px[1], 1.0])
    t = H @ p
    if abs(t[2]) < 1e-8:
        return None
    return (t[0] / t[2], t[1] / t[2])


def extraer_posicion_balon(tracks, frame_num):
    """
    Extrae la posición del balón en píxeles desde la estructura
    de tracks de tu pipeline.
    Ajustá esto según cómo se llamen tus categorías.
    """
    balon_data = tracks.get("ball", tracks.get("balon", {}))
    if not balon_data:
        return None

    if isinstance(balon_data, list):
        if frame_num >= len(balon_data):
            return None
        frame_tracks = balon_data[frame_num]
    else:
        frame_tracks = balon_data.get(frame_num, {})

    for _, item in frame_tracks.items():
        pos = item.get("position")
        if pos is not None:
            return pos
    return None


# ============ PRINCIPAL ============

def main(video_path, tracks=None):
    modelo = cargar_modelo()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")

    H_actual = None
    prev_balon_m = None
    prev_frame_num = -1
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Detectar keypoints y recalcular homografía cada N frames
        #    (podés hacerlo cada 5 o 10 frames para ir más rápido)
        if frame_num % 5 == 0:
            puntos_px = detectar_keypoints_cancha(modelo, frame)
            if puntos_px is not None:
                H_nueva = calcular_homografia(puntos_px, FIELD_POINTS_M)
                if H_nueva is not None:
                    H_actual = H_nueva

        # 2) Si tenemos homografía, calcular velocidad del balón
        if H_actual is not None:
            balon_px = None
            if tracks is not None:
                balon_px = extraer_posicion_balon(tracks, frame_num)

            if balon_px is not None:
                balon_m = pixel_a_metros(balon_px, H_actual)

                if balon_m is not None:
                    if prev_balon_m is not None and prev_frame_num >= 0:
                        dt = (frame_num - prev_frame_num) / fps
                        if dt > 0:
                            dx = balon_m[0] - prev_balon_m[0]
                            dy = balon_m[1] - prev_balon_m[1]
                            dist_m = np.sqrt(dx**2 + dy**2)
                            vel_mps = dist_m / dt
                            vel_kmh = vel_mps * 3.6

                            print(f"Frame {frame_num}: "
                                  f"{vel_kmh:.1f} km/h  ({vel_mps:.2f} m/s)")

                            # Dibujar sobre el frame
                            x, y = int(balon_px[0]), int(balon_px[1])
                            cv2.putText(frame, f"{vel_kmh:.1f} km/h",
                                        (x + 10, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 255, 255), 2, cv2.LINE_AA)

                    prev_balon_m = balon_m
                    prev_frame_num = frame_num

        # 3) Mostrar frame (opcional)
        cv2.imshow("Calibracion automatica", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Listo.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python auto_calibracion_velocidad.py <ruta_video>")
        sys.exit(1)

    # Si querés pasar tracks ya calculados, cambialo acá
    main(sys.argv[1], tracks=None)