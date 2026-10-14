"""
view_transformer_tramos.py

Carga calibracion_tramos.json y expone una interfaz compatible con
ViewTransformer de futbol_corregido.py, pero recalculando la homografia
en cada frame interpolando los pixel_vertices entre calibraciones.

Interfaz compatible:
  - enabled  (bool)
  - transform_point(point)  ->  np.array([x_m, y_m]) o None
  - add_transformed_position_to_tracks(tracks)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class InterpolatedViewTransformer:
    def __init__(self, config_path):
        self.enabled = False
        self.calibraciones = []          # lista de dicts con frame + puntos
        self.field_m = None
        self.total_frames = 0
        self._homografias_cache = {}     # frame -> matriz (cache)
        self._pixel_vertices_cache = {}  # frame -> np.array (4,2)

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"[Tramos] No se pudo cargar {config_path}: {exc}")
            return

        calibraciones = config.get("calibraciones", [])
        field_m = config.get("field_vertices_m")

        if not calibraciones or not field_m:
            print("[Tramos] JSON sin calibraciones o sin field_vertices_m.")
            return

        # Ordenar por frame
        calibraciones = sorted(calibraciones, key=lambda c: int(c["frame"]))

        self.calibraciones = [
            {
                "frame": int(c["frame"]),
                "pixel_vertices": np.asarray(
                    c["pixel_vertices"], dtype=np.float32
                ).reshape(4, 2),
            }
            for c in calibraciones
        ]
        self.field_m = np.asarray(field_m, dtype=np.float32).reshape(4, 2)
        self.total_frames = int(config.get("total_frames", 0))
        self.enabled = True

        print(
            f"[Tramos] {len(self.calibraciones)} calibraciones cargadas. "
            f"Frames: {self.calibraciones[0]['frame']} a "
            f"{self.calibraciones[-1]['frame']}."
        )

    def _pixel_vertices_en_frame(self, frame_num):
        """
        Devuelve los 4 pixel_vertices interpolados para este frame.
        Interpolacion lineal entre las dos calibraciones que lo rodean.
        Fuera del rango, usa el extremo mas cercano.
        """
        if not self.calibraciones:
            return None

        cal = self.calibraciones

        if frame_num <= cal[0]["frame"]:
            return cal[0]["pixel_vertices"]
        if frame_num >= cal[-1]["frame"]:
            return cal[-1]["pixel_vertices"]

        # Buscar el par que rodea al frame
        for i in range(len(cal) - 1):
            f0 = cal[i]["frame"]
            f1 = cal[i + 1]["frame"]
            if f0 <= frame_num <= f1:
                p0 = cal[i]["pixel_vertices"]
                p1 = cal[i + 1]["pixel_vertices"]
                if f1 == f0:
                    return p0
                t = (frame_num - f0) / (f1 - f0)
                return (1.0 - t) * p0 + t * p1

        return cal[-1]["pixel_vertices"]

    def _homografia_en_frame(self, frame_num):
        if frame_num in self._homografias_cache:
            return self._homografias_cache[frame_num]

        pixels = self._pixel_vertices_en_frame(frame_num)
        if pixels is None:
            return None

        try:
            H = cv2.getPerspectiveTransform(pixels, self.field_m)
        except cv2.error:
            return None

        if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
            return None

        self._homografias_cache[frame_num] = H
        self._pixel_vertices_cache[frame_num] = pixels
        return H

    def transform_point(self, point, frame_num=0):
        """
        Transforma un punto (x,y) en pixeles a metros.
        Necesita frame_num para elegir la homografia correcta.
        """
        if not self.enabled or point is None:
            return None

        p = np.asarray(point, dtype=np.float32).reshape(2)
        if not np.isfinite(p).all():
            return None

        H = self._homografia_en_frame(frame_num)
        if H is None:
            return None

        pixels = self._pixel_vertices_cache.get(frame_num)
        if pixels is None:
            pixels = self._pixel_vertices_en_frame(frame_num)

        inside = cv2.pointPolygonTest(
            pixels, (float(p[0]), float(p[1])), False
        )
        if inside < 0:
            return None

        transformed = cv2.perspectiveTransform(
            p.reshape(1, 1, 2), H
        ).reshape(2)

        if not np.isfinite(transformed).all():
            return None

        return transformed.astype(float)

    def add_transformed_position_to_tracks(self, tracks):
        """
        Recorre tracks y agrega position_transformed usando la homografia
        interpolada para cada frame.
        """
        if not self.enabled:
            # Si no hay calibracion, dejar todo en None para no romper nada
            for category, object_tracks in tracks.items():
                for frame_tracks in object_tracks:
                    for item in frame_tracks.values():
                        item["position_transformed"] = None
            return

        for category, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for item in frame_tracks.values():
                    pos = item.get("position")
                    item["position_transformed"] = self.transform_point(
                        pos, frame_num=frame_num
                    )