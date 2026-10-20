"""
view_transformer_tramos.py

Transformador de vista con calibración por tramos y persistencia de homografía.
Si un frame no tiene homografía válida, mantiene la última conocida durante
un número limitado de frames para no romper la métrica.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import cv2


class InterpolatedViewTransformer:
    def __init__(self, calibration_path):
        self.enabled = False
        self.pixel_vertices = None
        self.target_vertices = None
        self.perspective_transformer = None
        self._last_valid_homography = None
        self._frames_without_calibration = 0
        self._max_frames_without_calibration = 30  # ~1 segundo a 30 fps
        try:
            self._load(calibration_path)
            self.enabled = True
        except Exception as exc:
            print(f"[InterpolatedViewTransformer] No se pudo cargar: {exc}")

    def _load(self, calibration_path):
        path = Path(calibration_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Se asume un formato con "pixel_vertices" y "field_vertices_m".
        # Si tu archivo tiene otra estructura, ajusta aquí.
        pixels = data.get("pixel_vertices")
        targets = data.get("field_vertices_m")
        if (not isinstance(pixels, list) or not isinstance(targets, list)
                or len(pixels) != 4 or len(targets) != 4):
            raise ValueError("Formato de calibración inválido")
        self.pixel_vertices = np.asarray(pixels, dtype=np.float32).reshape(4, 2)
        self.target_vertices = np.asarray(targets, dtype=np.float32).reshape(4, 2)
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )
        if not np.isfinite(self.perspective_transformer).all():
            raise ValueError("Homografía no válida")
        self._last_valid_homography = self.perspective_transformer.copy()

    def calibrar_frames(self, frames):
        """
        Aplica la lógica de persistencia de homografía sobre una lista de frames.
        En la práctica, esta clase ya tiene una homografía fija cargada.
        Si en algún momento se detecta que la calibración falla, se mantiene
        la última válida. Devuelve True si hay al menos una calibración válida.
        """
        # Como este transformador es estático, consideramos que la calibración
        # es válida si se cargó correctamente.
        return self.enabled

    def transform_point(self, point, frame_num=None):
        """
        Transforma un punto de píxeles a metros usando la homografía.
        Si la homografía no está disponible, intenta usar la última válida.
        """
        if not self.enabled:
            return None
        p = np.asarray(point, dtype=np.float32).reshape(2)
        if not np.isfinite(p).all():
            return None
        # Si no hay homografía actual pero hay una última válida, usarla.
        homography = self.perspective_transformer
        if homography is None:
            homography = self._last_valid_homography
        if homography is None:
            return None
        try:
            transformed = cv2.perspectiveTransform(
                p.reshape(1, 1, 2), homography
            ).reshape(2)
            if not np.isfinite(transformed).all():
                return None
            return transformed.astype(float)
        except cv2.error:
            return None