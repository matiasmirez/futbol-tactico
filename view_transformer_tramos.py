"""Transformador métrico por tramos para calibracion_tramos.json.
Compatible con marcar_cancha_tramos.py y futbol_corregido.py.
"""
from pathlib import Path
import json
import cv2
import numpy as np


class InterpolatedViewTransformer:
    def __init__(self, calibration_path):
        self.calibration_path = Path(calibration_path)
        with self.calibration_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        self.enabled = False
        self.homografias = {}
        self._frames = []
        self._Hs = []
        self._segment_ids = []
        self.fps = float(data.get('fps') or 0)
        vertices_m = data.get('field_vertices_m')
        calibraciones = data.get('calibraciones')
        if not isinstance(vertices_m, list) or len(vertices_m) != 4:
            raise ValueError("field_vertices_m debe contener cuatro puntos.")
        if not isinstance(calibraciones, list) or not calibraciones:
            raise ValueError("calibraciones está vacío.")

        dst = np.asarray(vertices_m, dtype=np.float32).reshape(4, 2)
        prev_frame = -1
        for idx, cal in enumerate(calibraciones):
            frame = int(cal.get('frame', round(float(cal.get('time_seconds', 0)) * self.fps)))
            src = np.asarray(cal.get('pixel_vertices'), dtype=np.float32)
            if src.shape != (4, 2) or not np.isfinite(src).all():
                continue
            if frame <= prev_frame:
                raise ValueError("Los frames de calibración deben ser crecientes y únicos.")
            if abs(cv2.contourArea(src)) < 100 or not cv2.isContourConvex(src.astype(np.int32)):
                continue
            H = cv2.getPerspectiveTransform(src, dst)
            if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
                continue
            self._frames.append(frame)
            self._Hs.append(H)
            self._segment_ids.append(idx)
            self.homografias[frame] = H
            prev_frame = frame

        if not self._frames:
            raise ValueError("No hay homografías válidas en calibracion_tramos.json.")
        self.enabled = True

    def get_calibration_segment(self, frame_num=0):
        if not self._frames:
            return None
        frame = int(frame_num)
        # El identificador cambia en cada calibración para evitar velocidades
        # entre sistemas locales cuando la cámara cambia de posición.
        i = int(np.searchsorted(self._frames, frame, side='right') - 1)
        i = max(0, min(i, len(self._frames) - 1))
        return self._segment_ids[i]

    def _homography_for_frame(self, frame_num):
        frame = int(frame_num or 0)
        i = int(np.searchsorted(self._frames, frame, side='right') - 1)
        if i < 0:
            i = 0
        # Use nearest earlier calibration. Do not interpolate homography matrices:
        # that can produce unstable/projectively invalid transforms.
        return self._Hs[i]

    def transform_point(self, point, frame_num=0):
        if not self.enabled or point is None:
            return None
        p = np.asarray(point, dtype=np.float32).reshape(-1)
        if p.size < 2 or not np.isfinite(p[:2]).all():
            return None
        H = self._homography_for_frame(frame_num)
        out = cv2.perspectiveTransform(p[:2].reshape(1, 1, 2), H).reshape(2)
        if not np.isfinite(out).all():
            return None
        return out.astype(float)

    def add_transformed_position_to_tracks(self, tracks):
        for object_tracks in tracks.values():
            for frame_idx, frame_tracks in enumerate(object_tracks):
                for item in frame_tracks.values():
                    item['position_transformed'] = self.transform_point(
                        item.get('position'), frame_num=frame_idx
                    )
