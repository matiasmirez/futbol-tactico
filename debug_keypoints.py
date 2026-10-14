"""
debug_keypoints.py
Prueba el AutoCalibrador en un frame del video y muestra
qué keypoints detecta el modelo.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

VIDEO_PATH = Path(r"C:\Users\Matias\Downloads\chivas_penal.mp4")
MODEL_PATH = Path(r"C:\Users\Matias\Desktop\futbol\foot\modelos\Model\weights\best.pt")
FRAME_A_PROBAR = 105  # el frame que se ve en tu captura

# Índices de los 4 puntos de cada área
KEYPOINTS_AREAS = {
    "area_izquierda": [1, 2, 3, 4],
    "area_derecha":   [17, 18, 19, 20],
}

def main():
    print(f"Cargando modelo: {MODEL_PATH}")
    modelo = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_A_PROBAR)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("No se pudo leer el frame")
        return

    print(f"Frame {FRAME_A_PROBAR} leído: {frame.shape}")

    resultados = modelo(frame, verbose=False)
    r = resultados[0]

    if r.keypoints is None:
        print("❌ El modelo NO devolvió keypoints")
        return

    kpts = r.keypoints.data.cpu().numpy()
    if len(kpts.shape) == 3:
        kpts = kpts[0]

    print(f"✅ Keypoints detectados: {kpts.shape[0]}")
    print()

    # Mostrar TODOS los keypoints con su visibilidad
    print("Todos los keypoints (índice, x, y, visibilidad):")
    for i, (x, y, vis) in enumerate(kpts):
        marca = ""
        for nombre, indices in KEYPOINTS_AREAS.items():
            if i in indices:
                marca = f"  <-- {nombre}"
        print(f"  [{i:2d}] x={x:7.1f}  y={y:7.1f}  vis={vis:.3f}{marca}")

    print()

    # Probar las dos áreas
    for nombre, indices in KEYPOINTS_AREAS.items():
        print(f"--- {nombre} (índices {indices}) ---")
        ok = True
        for idx in indices:
            if idx >= len(kpts):
                print(f"  índice {idx} fuera de rango")
                ok = False
                continue
            x, y, vis = kpts[idx]
            estado = "OK" if vis >= 0.5 else "BAJA VIS"
            print(f"  [{idx}] x={x:.1f} y={y:.1f} vis={vis:.3f} {estado}")
            if vis < 0.5:
                ok = False

        if ok:
            print(f"  ✅ {nombre} se puede usar")
        else:
            print(f"  ❌ {nombre} NO se puede usar (visibilidad baja)")

    # Dibujar los keypoints sobre el frame para verlos
    vis_frame = frame.copy()
    for i, (x, y, vis) in enumerate(kpts):
        color = (0, 255, 0) if vis >= 0.5 else (0, 0, 255)
        cv2.circle(vis_frame, (int(x), int(y)), 5, color, -1)
        cv2.putText(vis_frame, str(i), (int(x)+6, int(y)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    salida = Path("debug_keypoints_frame.png")
    cv2.imwrite(str(salida), vis_frame)
    print(f"\nImagen guardada: {salida.resolve()}")


if __name__ == "__main__":
    main()