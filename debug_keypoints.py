
"""
debug_keypoints.py

Diagnóstico de los keypoints del modelo de pose utilizado por
auto_calibrador.py.

Analiza los fotogramas de:
    fotogramas_calibracion/frame_*.jpg

Genera:
    fotogramas_calibracion/diagnostico_keypoints/*.jpg
    fotogramas_calibracion/resumen_keypoints.csv

IMPORTANTE:
Este script diagnostica las detecciones. No crea una calibración
de cancha ni garantiza que los índices seleccionados correspondan
a las esquinas reales del área penal.
"""

from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CARPETA_FRAMES = BASE_DIR / "fotogramas_calibracion"

VIDEO_PATH = Path(
    r"C:\Users\Matias\Downloads\chivas_penal.mp4"
)

MODEL_PATH = (
    BASE_DIR / "modelos" / "Model" / "weights" / "best.pt"
)

CARPETA_SALIDA = CARPETA_FRAMES / "diagnostico_keypoints"

CSV_SALIDA = CARPETA_FRAMES / "resumen_keypoints.csv"

# Si hay imágenes frame_*.jpg, se analizan todas.
# Si la carpeta no contiene imágenes, se intenta leer el video.
FRAME_INICIAL_VIDEO = 0
PASO_VIDEO = 68

# Umbral orientativo, no garantía de precisión geométrica.
CONF_MIN = 0.15

# Grupos que usa actualmente auto_calibrador.py.
KEYPOINTS_AREAS = {
    "area_grupo_1": [1, 2, 3, 4],
    "area_grupo_2": [17, 18, 19, 20],
}


# ============================================================
# FUNCIONES
# ============================================================

def cargar_frames():
    """Devuelve una lista de (nombre, imagen, numero_frame)."""

    frames = []

    if CARPETA_FRAMES.exists():
        archivos = sorted(
            CARPETA_FRAMES.glob("frame_*.jpg")
        )

        for ruta in archivos:
            imagen = cv2.imread(str(ruta))

            if imagen is None:
                print(f"[AVISO] No se pudo abrir: {ruta.name}")
                continue

            try:
                numero = int(
                    ruta.stem.replace("frame_", "")
                )
            except ValueError:
                numero = -1

            frames.append((ruta.name, imagen, numero))

    if frames:
        print(
            f"[INFO] Se encontraron {len(frames)} "
            "fotogramas JPG."
        )
        return frames

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            "No hay imágenes frame_*.jpg y tampoco se "
            f"encuentra el video:\n{VIDEO_PATH}"
        )

    print("[INFO] No hay imágenes; leyendo el video.")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir el video: {VIDEO_PATH}"
        )

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    numero = FRAME_INICIAL_VIDEO

    while numero < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, numero)
        ret, imagen = cap.read()

        if ret and imagen is not None:
            nombre = f"frame_{numero:06d}.jpg"
            frames.append((nombre, imagen, numero))

        numero += PASO_VIDEO

    cap.release()

    print(f"[INFO] Frames extraídos para analizar: {len(frames)}")
    return frames


def obtener_deteccion_principal(resultado):
    """
    Selecciona la detección con mayor confianza de caja.
    Devuelve keypoints y confianzas, o (None, None).
    """

    if resultado.keypoints is None:
        return None, None

    if len(resultado.keypoints) == 0:
        return None, None

    kpts = resultado.keypoints

    xy = kpts.xy.cpu().numpy()

    if kpts.conf is not None:
        conf = kpts.conf.cpu().numpy()
    else:
        # Algunas representaciones incluyen x, y, visibilidad
        # dentro de keypoints.data. Si no, no inventamos confianza.
        data = kpts.data.cpu().numpy()
        if data.ndim == 3 and data.shape[2] >= 3:
            conf = data[:, :, 2]
        else:
            conf = np.zeros(xy.shape[:2], dtype=np.float32)

    if resultado.boxes is not None and len(resultado.boxes) == len(xy):
        conf_cajas = resultado.boxes.conf.cpu().numpy()
        indice = int(np.argmax(conf_cajas))
    else:
        indice = 0

    return xy[indice], conf[indice]


def analizar_grupo(xy, conf, indices, ancho, alto):
    """Evalúa confianza, límites y geometría aproximada del grupo."""

    puntos = []
    validos = []

    for idx in indices:
        if idx >= len(xy):
            puntos.append({
                "indice": idx,
                "x": None,
                "y": None,
                "conf": None,
                "valido": False,
            })
            continue

        x = float(xy[idx][0])
        y = float(xy[idx][1])
        c = float(conf[idx])

        dentro = (
            np.isfinite(x)
            and np.isfinite(y)
            and 0 <= x < ancho
            and 0 <= y < alto
        )

        valido = dentro and c >= CONF_MIN

        puntos.append({
            "indice": idx,
            "x": x,
            "y": y,
            "conf": c,
            "valido": bool(valido),
        })

        if valido:
            validos.append((idx, x, y))

    area = None
    lados = None
    orden_coherente = False

    if len(validos) == 4:
        # Conserva el orden de los índices configurados.
        coords = np.array(
            [[p[1], p[2]] for p in validos],
            dtype=np.float32,
        )

        area = float(abs(cv2.contourArea(coords)))

        lados = [
            float(
                np.linalg.norm(
                    coords[(i + 1) % 4] - coords[i]
                )
            )
            for i in range(4)
        ]

        # Comprueba que los lados no colapsen a longitud cero.
        orden_coherente = (
            area >= 10.0
            and all(lado >= 3.0 for lado in lados)
            and cv2.isContourConvex(
                coords.reshape(-1, 1, 2)
            )
        )

    return {
        "puntos": puntos,
        "cantidad_validos": len(validos),
        "area_px2": area,
        "lados_px": lados,
        "orden_coherente": orden_coherente,
    }


def dibujar_resultado(imagen, xy, conf, grupos, numero_frame):
    """Dibuja keypoints y destaca los grupos de interés."""

    salida = imagen.copy()

    colores_grupo = {
        "area_grupo_1": (255, 180, 0),
        "area_grupo_2": (255, 0, 255),
    }

    indices_grupo = {
        idx
        for indices in KEYPOINTS_AREAS.values()
        for idx in indices
    }

    # Dibuja todos los keypoints disponibles.
    for idx, (punto, confianza) in enumerate(zip(xy, conf)):
        x = float(punto[0])
        y = float(punto[1])
        c = float(confianza)

        if not (
            np.isfinite(x)
            and np.isfinite(y)
            and 0 <= x < imagen.shape[1]
            and 0 <= y < imagen.shape[0]
        ):
            continue

        centro = (int(round(x)), int(round(y)))

        if c >= CONF_MIN:
            color = (0, 220, 0)
        else:
            color = (0, 0, 255)

        radio = 7 if idx in indices_grupo else 4

        cv2.circle(salida, centro, radio, color, -1)

        etiqueta = f"{idx}:{c:.2f}"

        cv2.putText(
            salida,
            etiqueta,
            (centro[0] + 6, centro[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    # Dibuja los grupos únicamente si los cuatro puntos
    # superan el umbral y están dentro de la imagen.
    for nombre, indices in KEYPOINTS_AREAS.items():
        grupo = grupos[nombre]

        if grupo["cantidad_validos"] != 4:
            continue

        puntos = grupo["puntos"]

        coords = np.array(
            [
                [p["x"], p["y"]]
                for p in puntos
            ],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        color = colores_grupo[nombre]

        cv2.polylines(
            salida,
            [coords],
            isClosed=True,
            color=color,
            thickness=2,
        )

    cv2.putText(
        salida,
        f"Frame {numero_frame}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return salida


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra el modelo de pose:\n{MODEL_PATH}"
        )

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Modelo: {MODEL_PATH}")
    print(f"[INFO] Umbral de confianza: {CONF_MIN}")
    print("[INFO] Cargando YOLO...")

    modelo = YOLO(str(MODEL_PATH))

    frames = cargar_frames()

    if not frames:
        raise RuntimeError("No hay fotogramas que analizar.")

    filas_csv = []

    for nombre, imagen, numero_frame in frames:
        alto, ancho = imagen.shape[:2]

        print("\n" + "=" * 72)
        print(
            f"ARCHIVO: {nombre} | frame={numero_frame} "
            f"| resolución={ancho}x{alto}"
        )

        resultado = modelo.predict(
            source=imagen,
            conf=0.01,
            verbose=False,
        )[0]

        xy, conf = obtener_deteccion_principal(resultado)

        if xy is None:
            print("[SIN DETECCIÓN] El modelo no devolvió keypoints.")

            filas_csv.append({
                "archivo": nombre,
                "frame": numero_frame,
                "deteccion": False,
                "grupo": "",
                "puntos_validos": 0,
                "area_px2": "",
                "lados_px": "",
                "orden_coherente": False,
            })
            continue

        print(f"Keypoints detectados: {len(xy)}")
        print("\nTodos los keypoints:")

        for idx, (punto, confianza) in enumerate(zip(xy, conf)):
            x, y = map(float, punto)
            c = float(confianza)

            marcas = [
                grupo
                for grupo, indices in KEYPOINTS_AREAS.items()
                if idx in indices
            ]

            marca = f" <-- {', '.join(marcas)}" if marcas else ""

            print(
                f"  [{idx:02d}] x={x:8.1f} "
                f"y={y:8.1f} conf={c:.3f}{marca}"
            )

        grupos = {}

        for nombre_grupo, indices in KEYPOINTS_AREAS.items():
            datos = analizar_grupo(
                xy, conf, indices, ancho, alto
            )

            grupos[nombre_grupo] = datos

            print(f"\n--- {nombre_grupo}: índices {indices} ---")

            for punto in datos["puntos"]:
                if punto["conf"] is None:
                    print(
                        f"  kp {punto['indice']}: fuera de rango"
                    )
                    continue

                print(
                    f"  kp {punto['indice']:02d}: "
                    f"x={punto['x']:.1f}, "
                    f"y={punto['y']:.1f}, "
                    f"conf={punto['conf']:.3f}, "
                    f"válido={punto['valido']}"
                )

            print(
                f"  Puntos válidos: "
                f"{datos['cantidad_validos']}/4"
            )

            if datos["area_px2"] is not None:
                print(
                    f"  Área del cuadrilátero: "
                    f"{datos['area_px2']:.1f} px²"
                )
                print(
                    "  Lados consecutivos: "
                    + ", ".join(
                        f"{lado:.1f} px"
                        for lado in datos["lados_px"]
                    )
                )

            print(
                f"  Geometría convexa y no degenerada: "
                f"{datos['orden_coherente']}"
            )

            filas_csv.append({
                "archivo": nombre,
                "frame": numero_frame,
                "deteccion": True,
                "grupo": nombre_grupo,
                "puntos_validos": datos["cantidad_validos"],
                "area_px2": (
                    datos["area_px2"]
                    if datos["area_px2"] is not None
                    else ""
                ),
                "lados_px": (
                    ";".join(
                        f"{lado:.2f}"
                        for lado in datos["lados_px"]
                    )
                    if datos["lados_px"] is not None
                    else ""
                ),
                "orden_coherente": datos["orden_coherente"],
            })

        anotada = dibujar_resultado(
            imagen,
            xy,
            conf,
            grupos,
            numero_frame,
        )

        destino = CARPETA_SALIDA / nombre

        if not cv2.imwrite(str(destino), anotada):
            print(f"[AVISO] No se pudo guardar: {destino}")
        else:
            print(f"\n[IMAGEN] {destino}")

    columnas = [
        "archivo",
        "frame",
        "deteccion",
        "grupo",
        "puntos_validos",
        "area_px2",
        "lados_px",
        "orden_coherente",
    ]

    with CSV_SALIDA.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=columnas,
        )
        escritor.writeheader()
        escritor.writerows(filas_csv)

    print("\n" + "=" * 72)
    print("[FINALIZADO]")
    print(f"Imágenes anotadas: {CARPETA_SALIDA}")
    print(f"Resumen CSV: {CSV_SALIDA}")
    print(
        "\nATENCIÓN: confianza y geometría no prueban que "
        "los keypoints sean esquinas reales de la cancha."
    )


if __name__ == "__main__":
    main()
