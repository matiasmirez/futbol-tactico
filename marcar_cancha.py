"""Calibración manual de un rectángulo visible de cancha.

Uso:
  python marcar_cancha.py VIDEO [FRAME] [ANCHO_M] [ALTO_M]

Seleccioná 4 esquinas del MISMO rectángulo, en orden:
  superior-izquierda, superior-derecha, inferior-derecha, inferior-izquierda.

El rectángulo puede ser el área grande (40.32 x 16.5), el área chica
(18.32 x 5.5) u otro rectángulo cuyas dimensiones reales conozcas.
"""
import sys
import json
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = BASE_DIR / "calibracion_cancha.json"


def extraer_frame(video_path, frame_num):
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"No se pudo leer el frame {frame_num}")
        return frame
    finally:
        cap.release()


def validar_puntos(puntos, ancho_m, alto_m):
    p = np.asarray(puntos, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(p).all():
        raise ValueError("Los puntos contienen coordenadas inválidas.")
    if abs(cv2.contourArea(p)) < 100:
        raise ValueError("Los cuatro puntos forman un área demasiado pequeña.")
    if not np.isfinite([ancho_m, alto_m]).all() or ancho_m <= 0 or alto_m <= 0:
        raise ValueError("Ancho y alto deben ser números positivos.")
    # Los puntos deben formar un cuadrilátero convexo, sin cruces.
    if not cv2.isContourConvex(p.astype(np.int32)):
        raise ValueError("Los puntos no forman un cuadrilátero convexo. Revisá el orden.")
    return p.astype(int).tolist()


def pedir_puntos(frame):
    puntos = []
    ventana = "Marcar 4 esquinas | u=deshacer r=reset ENTER=confirmar ESC=salir"

    def dibujar():
        img = frame.copy()
        for i, punto in enumerate(puntos):
            cv2.circle(img, punto, 6, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (punto[0] + 8, punto[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        for i in range(1, len(puntos)):
            cv2.line(img, puntos[i - 1], puntos[i], (0, 255, 0), 2)
        if len(puntos) == 4:
            cv2.line(img, puntos[3], puntos[0], (0, 255, 0), 2)
        cv2.putText(img, f"Puntos: {len(puntos)}/4", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        return img

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
            puntos.append((int(x), int(y)))

    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(ventana, click)
    try:
        while True:
            cv2.imshow(ventana, dibujar())
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                return None
            if key == ord("u") and puntos:
                puntos.pop()
            elif key == ord("r"):
                puntos.clear()
            elif key in (10, 13, 32) and len(puntos) == 4:
                return list(puntos)
    finally:
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Uso: python marcar_cancha.py VIDEO [FRAME] [ANCHO_M] [ALTO_M]")
        raise SystemExit(2)
    video = Path(sys.argv[1])
    frame_num = int(sys.argv[2]) if len(sys.argv) > 2 else 105
    ancho_m = float(sys.argv[3]) if len(sys.argv) > 3 else 40.32
    alto_m = float(sys.argv[4]) if len(sys.argv) > 4 else 16.5
    if not video.exists():
        raise SystemExit(f"No existe el video: {video}")
    if ancho_m <= 0 or alto_m <= 0:
        raise SystemExit("Las dimensiones deben ser positivas.")

    frame = extraer_frame(video, frame_num)
    print("Marcá las esquinas del mismo rectángulo, en orden TL, TR, BR, BL.")
    print(f"Dimensiones declaradas: ancho={ancho_m:.3f} m, alto={alto_m:.3f} m")
    print("No uses dimensiones del área grande si los puntos son de otra marca.")
    puntos = pedir_puntos(frame)
    if puntos is None:
        print("Cancelado; no se modificó la calibración.")
        return
    try:
        puntos = validar_puntos(puntos, ancho_m, alto_m)
    except ValueError as exc:
        raise SystemExit(str(exc))

    destino = [[0.0, 0.0], [ancho_m, 0.0],
               [ancho_m, alto_m], [0.0, alto_m]]
    data = {
        "version": 2,
        "tipo": "calibracion_estatica",
        "video": video.name,
        "fps_frame": frame_num,
        "pixel_vertices": puntos,
        "field_vertices_m": destino,
        "ancho_m": ancho_m,
        "alto_m": alto_m,
        "orden_puntos": ["superior_izquierda", "superior_derecha",
                         "inferior_derecha", "inferior_izquierda"],
        "nota": "Válida solo mientras no cambie el encuadre/zoom. Verificar visualmente.",
    }
    tmp = CALIBRATION_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CALIBRATION_PATH)
    print(f"Calibración guardada: {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
