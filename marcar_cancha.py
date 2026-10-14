"""
marcar_cancha.py

Abre un frame del video, te deja clickear 4 puntos de la cancha
y guarda calibracion_cancha.json listo para usar.

Orden de clicks (importante):
  1) Esquina superior-izquierda del area (o del rectangulo que elijas)
  2) Esquina superior-derecha
  3) Esquina inferior-derecha
  4) Esquina inferior-izquierda

Uso:
    python marcar_cancha.py <ruta_video> [frame_numero]

Ejemplo:
    python marcar_cancha.py "C:\\Users\\Matias\\Downloads\\chivas_penal.mp4" 105
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = BASE_DIR / "calibracion_cancha.json"


def extraer_frame(video_path, frame_num):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"No se pudo leer el frame {frame_num}")

    return frame


def pedir_puntos(frame):
    """
    Abre una ventana y deja clickear 4 puntos.
    Teclas:
      u  -> deshacer ultimo punto
      r  -> resetear
      ENTER o ESPACIO -> confirmar (solo si hay 4)
      ESC -> cancelar
    """
    puntos = []
    ventana = "marca 4 puntos - u=deshacer, r=reset, ENTER=ok, ESC=cancelar"

    def redibujar():
        img = frame.copy()

        for i, (x, y) in enumerate(puntos):
            cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                img,
                f"{i+1}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if len(puntos) >= 2:
            for i in range(len(puntos) - 1):
                cv2.line(img, puntos[i], puntos[i + 1], (0, 255, 0), 2)

        if len(puntos) == 4:
            cv2.line(img, puntos[3], puntos[0], (0, 255, 0), 2)

        cv2.putText(
            img,
            f"Puntos: {len(puntos)}/4",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return img

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
            puntos.append((x, y))

    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(ventana, on_click)

    while True:
        cv2.imshow(ventana, redibujar())
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return None

        if key == ord("u") and puntos:
            puntos.pop()

        if key == ord("r"):
            puntos.clear()

        if key in (13, 32) and len(puntos) == 4:  # ENTER o ESPACIO
            cv2.destroyAllWindows()
            return puntos

    return None


def pedir_metros():
    """
    Pide las coordenadas reales en metros de esos 4 puntos.
    Por defecto asume que marcaste las 4 esquinas de un rectangulo
    y te pide ancho y alto.
    """
    print("\nAhora decime las medidas reales del rectangulo que marcaste.")
    print("Ejemplo tipico: area grande = 16.5 m de fondo x 40.3 m de ancho")
    print("Area chica = 5.5 m de fondo x 18.32 m de ancho")
    print("(si marcaste otro rectangulo, pone las medidas que correspondan)\n")

    ancho = input("Ancho real del rectangulo en metros [40.3]: ").strip()
    alto = input("Alto real del rectangulo en metros [16.5]: ").strip()

    try:
        ancho = float(ancho) if ancho else 40.3
    except ValueError:
        ancho = 40.3

    try:
        alto = float(alto) if alto else 16.5
    except ValueError:
        alto = 16.5

    return ancho, alto


def main():
    if len(sys.argv) < 2:
        print("Uso: python marcar_cancha.py <ruta_video> [frame_numero]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    frame_num = int(sys.argv[2]) if len(sys.argv) > 2 else 105

    if not video_path.exists():
        print(f"No existe el video: {video_path}")
        sys.exit(1)

    print(f"Extrayendo frame {frame_num} de {video_path.name}...")
    frame = extraer_frame(video_path, frame_num)

    print("\nMarça 4 puntos en este orden:")
    print("  1) esquina superior-izquierda")
    print("  2) esquina superior-derecha")
    print("  3) esquina inferior-derecha")
    print("  4) esquina inferior-izquierda")
    print("\nTeclas: u=deshacer, r=reset, ENTER=confirmar, ESC=cancelar\n")

    puntos = pedir_puntos(frame)

    if not puntos or len(puntos) != 4:
        print("Cancelado.")
        sys.exit(0)

    ancho_m, alto_m = pedir_metros()

    # Orden: TL, TR, BR, BL
    field_vertices_m = [
        [0.0, 0.0],
        [ancho_m, 0.0],
        [ancho_m, alto_m],
        [0.0, alto_m],
    ]

    data = {
        "instrucciones": (
            "Configuración generada por marcar_cancha.py. "
            "Los pixel_vertices y field_vertices_m están en el mismo orden. "
            "Válida solo para frames donde la cámara no se haya movido mucho."
        ),
        "pixel_vertices": [[int(x), int(y)] for x, y in puntos],
        "field_vertices_m": field_vertices_m,
        "frame_de_calibracion": frame_num,
        "nota": (
            "Una homografía estática deja de ser válida cuando la cámara "
            "panea o hace zoom. Recalibrar por tramos si es necesario."
        ),
    }

    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado: {CALIBRATION_PATH}")
    print("pixel_vertices:", data["pixel_vertices"])
    print("field_vertices_m:", data["field_vertices_m"])


if __name__ == "__main__":
    main()