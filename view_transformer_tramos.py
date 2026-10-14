
"""
view_transformer_tramos.py

Transforma coordenadas de píxeles a metros usando calibraciones
guardadas en calibracion_tramos.json.

Interpola los cuatro vértices de la cancha entre calibraciones
conocidas y recalcula la homografía para cada frame.

Interfaz compatible con futbol_corregido.py:
    enabled
    transform_point(point, frame_num=0)
    add_transformed_position_to_tracks(tracks)
"""

from pathlib import Path
import json

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

FIELD_POINTS_DEFAULT = np.array(
    [
        [0.0, 0.0],
        [105.0, 0.0],
        [105.0, 68.0],
        [0.0, 68.0],
    ],
    dtype=np.float32,
)


class InterpolatedViewTransformer:
    def __init__(self, calibration_path=None):
        self.enabled = False
        self.calibration_path = Path(
            calibration_path or BASE_DIR / "calibracion_tramos.json"
        )

        self.calibraciones = []
        self.field_vertices_m = FIELD_POINTS_DEFAULT.copy()
        self._cache = {}

        self._cargar()

    def _cargar(self):
        if not self.calibration_path.exists():
            print(
                "[Calibración por tramos] No existe: "
                f"{self.calibration_path}"
            )
            return

        try:
            with self.calibration_path.open(
                "r", encoding="utf-8"
            ) as archivo:
                data = json.load(archivo)

            if not isinstance(data, dict):
                raise ValueError(
                    "El JSON debe contener un objeto."
                )

            field = data.get(
                "field_vertices_m",
                data.get(
                    "field_points_m",
                    data.get("field_vertices", FIELD_POINTS_DEFAULT.tolist()),
                ),
            )

            field = np.asarray(field, dtype=np.float32)

            if field.shape != (4, 2) or not np.isfinite(field).all():
                raise ValueError(
                    "'field_vertices_m' debe tener cuatro puntos [x, y]."
                )

            self.field_vertices_m = field

            # Admite distintos nombres para la lista de calibraciones.
            lista = None
            for clave in (
                "tramos",
                "calibraciones",
                "keyframes",
                "calibration_points",
                "frames",
            ):
                if isinstance(data.get(clave), list):
                    lista = data[clave]
                    break

            # También permite que el JSON contenga directamente
            # una lista de calibraciones.
            if lista is None and isinstance(data, list):
                lista = data

            if not lista:
                raise ValueError(
                    "No se encontró una lista de calibraciones "
                    "(tramos o calibraciones)."
                )

            calibraciones = []

            for item in lista:
                if not isinstance(item, dict):
                    continue

                frame = item.get(
                    "frame_num",
                    item.get(
                        "frame",
                        item.get(
                            "numero_frame",
                            item.get("n_frame"),
                        ),
                    ),
                )

                pixels = item.get(
                    "pixel_vertices",
                    item.get(
                        "pixels",
                        item.get("vertices_px"),
                    ),
                )

                if frame is None or pixels is None:
                    continue

                pixels = np.asarray(pixels, dtype=np.float32)

                if (
                    pixels.shape != (4, 2)
                    or not np.isfinite(pixels).all()
                ):
                    continue

                # Cada punto debe ser distinto.
                if len(np.unique(pixels, axis=0)) != 4:
                    continue

                calibraciones.append(
                    {
                        "frame": int(frame),
                        "pixels": pixels,
                    }
                )

            calibraciones.sort(key=lambda c: c["frame"])

            if not calibraciones:
                raise ValueError(
                    "No hay calibraciones válidas con frame y "
                    "cuatro pixel_vertices."
                )

            # Si hay varios registros del mismo frame, conserva el último.
            unicas = {}
            for calibracion in calibraciones:
                unicas[calibracion["frame"]] = calibracion

            self.calibraciones = sorted(
                unicas.values(),
                key=lambda c: c["frame"],
            )

            self._cache.clear()
            self.enabled = True

            print(
                "[Calibración por tramos] Cargadas "
                f"{len(self.calibraciones)} calibraciones."
            )
            print(
                "[Calibración por tramos] Rango de frames: "
                f"{self.calibraciones[0]['frame']}–"
                f"{self.calibraciones[-1]['frame']}."
            )

        except (OSError, ValueError, TypeError, KeyError,
                json.JSONDecodeError) as exc:
            self.enabled = False
            print(
                "[Calibración por tramos] Error leyendo JSON: "
                f"{exc}"
            )

    def _vertices_para_frame(self, frame_num):
        if not self.calibraciones:
            return None

        frame_num = int(frame_num)

        if frame_num in self._cache:
            return self._cache[frame_num][0]

        # Antes de la primera calibración: usar la primera.
        if frame_num <= self.calibraciones[0]["frame"]:
            return self.calibraciones[0]["pixels"].copy()

        # Después de la última: usar la última.
        if frame_num >= self.calibraciones[-1]["frame"]:
            return self.calibraciones[-1]["pixels"].copy()

        for i in range(len(self.calibraciones) - 1):
            a = self.calibraciones[i]
            b = self.calibraciones[i + 1]

            if a["frame"] <= frame_num <= b["frame"]:
                intervalo = b["frame"] - a["frame"]

                if intervalo <= 0:
                    return a["pixels"].copy()

                alpha = (
                    (frame_num - a["frame"]) / intervalo
                )

                vertices = (
                    a["pixels"] * (1.0 - alpha)
                    + b["pixels"] * alpha
                )

                return vertices.astype(np.float32)

        return None

    def _homografia_para_frame(self, frame_num):
        frame_num = int(frame_num)

        if frame_num in self._cache:
            return self._cache[frame_num][1]

        pixels = self._vertices_para_frame(frame_num)

        if pixels is None:
            return None

        # Rechazar cuadriláteros degenerados.
        area = abs(cv2.contourArea(pixels))

        if area < 10.0:
            return None

        H = cv2.getPerspectiveTransform(
            pixels.astype(np.float32),
            self.field_vertices_m.astype(np.float32),
        )

        if not np.isfinite(H).all():
            return None

        if abs(float(np.linalg.det(H))) < 1e-12:
            return None

        self._cache[frame_num] = (pixels, H)
        return H

    def transform_point(self, point, frame_num=0):
        if not self.enabled or point is None:
            return None

        try:
            p = np.asarray(
                point, dtype=np.float32
            ).reshape(-1)

            if p.size < 2 or not np.isfinite(p[:2]).all():
                return None

            H = self._homografia_para_frame(frame_num)

            if H is None:
                return None

            resultado = cv2.perspectiveTransform(
                p[:2].reshape(1, 1, 2),
                H,
            ).reshape(2)

            if not np.isfinite(resultado).all():
                return None

            return resultado.astype(float)

        except (ValueError, TypeError, cv2.error):
            return None

    def add_transformed_position_to_tracks(self, tracks):
        if not isinstance(tracks, dict):
            return

        for _, object_tracks in tracks.items():
            if not isinstance(object_tracks, (list, tuple)):
                continue

            for frame_num, frame_tracks in enumerate(object_tracks):
                if not isinstance(frame_tracks, dict):
                    continue

                for item in frame_tracks.values():
                    if not isinstance(item, dict):
                        continue

                    item["position_transformed"] = (
                        self.transform_point(
                            item.get("position"),
                            frame_num=frame_num,
                        )
                    )