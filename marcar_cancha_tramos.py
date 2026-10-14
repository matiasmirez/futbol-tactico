"""
marcar_cancha_tramos.py

Marca 4 puntos de la cancha cada X segundos para calibrar un video
donde la camara se mueve (paneo/zoom). Guarda calibracion_tramos.json
con una lista de calibraciones y sus frames.

Uso:
    python marcar_cancha_tramos.py <ruta_video> [cada_segundos] [ancho_m] [alto_m]

Ejemplo:
    python marcar_cancha_tramos.py "C:\\Users\\Matias\\Downloads\\chivas_penal.mp4" 2.0 40.3 16.5

Orden de clicks (igual que antes):
  1) esquina superior-izquierda
  2) esquina superior-derecha
  3) esquina inferior-derecha
  4) esquina inferior-izquierda

Teclas:
  u  -> deshacer ultimo punto
  r  -> resetear el tramo actual
  ENTER o ESPACIO -> confirmar tramo
  s  -> saltar este tramo
  q  -> terminar y guardar lo marcado
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SALIDA = BASE_DIR / "calibracion_tramos.json"


def info_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    return float(fps), total


def leer_frame(cap, frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
    ret, frame = cap.read()
    return frame if ret else None


def pedir_puntos(frame, titulo):
    puntos = []
    ventana = titulo

    def redibujar():
        img = frame.copy()
        for i, (x, y) in enumerate(puntos):
            cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(img, f"{i+1}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
        if len(puntos) >= 2:
            for i in range(len(puntos) - 1):
                cv2.line(img, puntos[i], puntos[i+1], (0, 255, 0), 2)
        if len(puntos) == 4:
            cv2.line(img, puntos[3], puntos[0], (0, 255, 0), 2)
        cv2.putText(img, f"Puntos: {len(puntos)}/4  (u=undo r=reset ENTER=ok s=saltar q=fin)",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2, cv2.LINE_AA)
        return img

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
            puntos.append((x, y))

    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(ventana, on_click)

    while True:
        cv2.imshow(ventana, redibujar())
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            return "quit"
        if key == ord("s"):
            cv2.destroyAllWindows()
            return "skip"
        if key == ord("u") and puntos:
            puntos.pop()
        if key == ord("r"):
            puntos.clear()
        if key in (13, 32) and len(puntos) == 4:
            cv2.destroyAllWindows()
            return list(puntos)


def main():
    if len(sys.argv) < 2:
        print("Uso: python marcar_cancha_tramos.py <video> [cada_seg] [ancho_m] [alto_m]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    cada_seg = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    ancho_m = float(sys.argv[3]) if len(sys.argv) > 3 else 40.3
    alto_m = float(sys.argv[4]) if len(sys.argv) > 4 else 16.5

    if not video_path.exists():
        print(f"No existe: {video_path}")
        sys.exit(1)

    fps, total = info_video(video_path)
    print(f"Video: {video_path.name}")
    print(f"FPS: {fps:.2f}  Frames: {total}  Duracion: {total/fps:.1f}s")
    print(f"Marca cada {cada_seg:.2f}s ({int(cada_seg*fps)} frames)")
    print(f"Rectangulo: {ancho_m} m x {alto_m} m\n")

    field_m = [
        [0.0, 0.0],
        [ancho_m, 0.0],
        [ancho_m, alto_m],
        [0.0, alto_m],
    ]

    cap = cv2.VideoCapture(str(video_path))
    calibraciones = []

    frame_objetivo = 0
    indice = 0

    while frame_objetivo < total:
        frame = leer_frame(cap, frame_objetivo)
        if frame is None:
            break

        seg_actual = frame_objetivo / fps
        titulo = f"Tramo {indice+1} | t={seg_actual:.2f}s | frame {frame_objetivo}"

        print(f"\n[{indice+1}] frame {frame_objetivo}  t={seg_actual:.2f}s")
        resultado = pedir_puntos(frame, titulo)

        if resultado == "quit":
            print("Terminado por el usuario.")
            break
        if resultado == "skip":
            print("Saltado.")
            frame_objetivo += int(cada_seg * fps)
            indice += 1
            continue

        calibraciones.append({
            "frame": int(frame_objetivo),
            "time_seconds": round(seg_actual, 3),
            "pixel_vertices": [[int(x), int(y)] for x, y in resultado],
        })
        print(f"  Guardado: {resultado}")

        frame_objetivo += int(cada_seg * fps)
        indice += 1

    cap.release()

    if not calibraciones:
        print("\nNo se marco ninguna calibracion.")
        return

    data = {
        "instrucciones": (
            "Calibraciones por tramos. Cada entrada tiene un frame y 4 puntos "
            "en pixeles. El pipeline interpola la homografia entre tramos. "
            "field_vertices_m es comun a todos los tramos."
        ),
        "fps": fps,
        "total_frames": total,
        "ancho_m": ancho_m,
        "alto_m": alto_m,
        "field_vertices_m": field_m,
        "calibraciones": calibraciones,
        "nota": (
            "Si el paneo es muy fuerte, marcar cada 1s. Si la camara esta "
            "casi quieta, con 2-3 calibraciones alcanza."
        ),
    }

    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado: {SALIDA}")
    print(f"Calibraciones: {len(calibraciones)}")
    for c in calibraciones:
        print(f"  frame {c['frame']} ({c['time_seconds']}s): {c['pixel_vertices']}")


if __name__ == "__main__":
    main()