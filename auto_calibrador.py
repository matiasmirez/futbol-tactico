"""
auto_calibrador.py

Calibrador automático con Soccana_Keypoint + optical flow.

Estrategia:
  1. Recorre el video buscando el frame donde la visibilidad de los
     4 puntos del área grande sea máxima Y formen un rectángulo coherente.
  2. En ese frame calcula la homografía inicial.
  3. Propaga los 4 puntos al resto del video con optical flow,
     recalculando la homografía en cada frame.

Interfaz compatible con el pipeline:
  - enabled
  - transform_point(point, frame_num=0)
  - add_transformed_position_to_tracks(tracks)
  - calibrar_frames(frames)
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ============ CONFIGURACIÓN ============

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "modelos" / "Model" / "weights" / "best.pt"
)

ANCHO_AREA_M = 40.32
ALTO_AREA_M = 16.5

FIELD_POINTS_M = np.array([
    [0.0,           0.0],
    [ANCHO_AREA_M,  0.0],
    [ANCHO_AREA_M,  ALTO_AREA_M],
    [0.0,           ALTO_AREA_M],
], dtype=np.float32)

# Índices de los 4 puntos del área grande
KEYPOINTS_AREAS = [
    [1, 2, 3, 4],       # área izquierda
    [17, 18, 19, 20],   # área derecha
]

# Visibilidad mínima para aceptar un keypoint en la búsqueda del mejor frame
VISIBILIDAD_MIN_BUSQUEDA = 0.15

# Cada cuántos frames revisar durante la búsqueda del mejor frame
PASO_BUSQUEDA = 3

# Parámetros de optical flow
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# --- Validación geométrica del área detectada ---
# Relación de aspecto esperada del área (ancho / alto en píxeles).
# Para una vista aérea perfecta sería ANCHO_AREA_M / ALTO_AREA_M = 2.44.
# En una vista de cámara de fútbol real, suele estar entre 1.5 y 6.0.
ASPECTO_MIN = 1.2
ASPECTO_MAX = 8.0

# Área mínima del polígono en píxeles (para descartar detecciones basura)
AREA_MIN_PX = 5000.0


class AutoCalibrador:
    def __init__(self, model_path=MODEL_PATH):
        self.enabled = False
        self.modelo = None
        self.homografias = {}
        self.pixel_vertices = {}
        self.field_m = FIELD_POINTS_M
        self.mejor_frame = None
        self.mejores_puntos = None

        if not Path(model_path).exists():
            print(f"[AutoCalibrador] No se encontró el modelo: {model_path}")
            return

        try:
            self.modelo = YOLO(str(model_path))
            self.enabled = True
            print(f"[AutoCalibrador] Modelo cargado: {model_path.name}")
        except Exception as exc:
            print(f"[AutoCalibrador] Error al cargar modelo: {exc}")

    # ---------------------------------------------------------------
    # VALIDACIÓN GEOMÉTRICA
    # ---------------------------------------------------------------
    @staticmethod
    def _area_es_coherente(puntos_px):
        """
        Verifica que los 4 puntos formen un rectángulo razonable.
        Devuelve True si pasa la validación.
        """
        if puntos_px is None or len(puntos_px) != 4:
            return False

        pts = np.asarray(puntos_px, dtype=np.float32)
        if not np.isfinite(pts).all():
            return False

        # Área del polígono
        area = abs(cv2.contourArea(pts))
        if area < AREA_MIN_PX:
            return False

        # Lados: 0->1 (ancho), 1->2 (alto), 2->3 (ancho), 3->0 (alto)
        lados = [
            np.linalg.norm(pts[1] - pts[0]),
            np.linalg.norm(pts[2] - pts[1]),
            np.linalg.norm(pts[3] - pts[2]),
            np.linalg.norm(pts[0] - pts[3]),
        ]

        if min(lados) < 1.0:
            return False

        ancho_px = (lados[0] + lados[2]) / 2.0
        alto_px = (lados[1] + lados[3]) / 2.0

        if alto_px < 1.0:
            return False

        aspecto = ancho_px / alto_px
        if not (ASPECTO_MIN <= aspecto <= ASPECTO_MAX):
            return False

        # Verificar que no haya auto-intersección (polígono simple)
        # Si el polígono se cruza, cv2.contourArea da un valor pequeño
        # comparado con el área del bounding box.
        x, y, w, h = cv2.boundingRect(pts)
        bbox_area = float(w * h)
        if bbox_area > 0 and area / bbox_area < 0.5:
            # El polígono ocupa menos del 50% del bounding box:
            # o está muy inclinado, o se auto-intersecta.
            return False

        return True

    # ---------------------------------------------------------------
    # BÚSQUEDA DEL MEJOR FRAME
    # ---------------------------------------------------------------
    def _leer_keypoints(self, frame):
        resultados = self.modelo(frame, verbose=False)
        if not resultados or resultados[0].keypoints is None:
            return None
        kpts = resultados[0].keypoints.data.cpu().numpy()
        if len(kpts.shape) == 3:
            kpts = kpts[0]
        return kpts

    def _visibilidad_area(self, kpts):
        """
        Devuelve (score, puntos, indices) del área con mayor visibilidad
        que además pase la validación geométrica.
        """
        mejor_score = -1.0
        mejores_puntos = None
        mejores_indices = None

        for indices in KEYPOINTS_AREAS:
            visibilidades = []
            puntos = []
            valido = True

            for idx in indices:
                if idx >= len(kpts):
                    valido = False
                    break
                x, y, vis = kpts[idx]
                visibilidades.append(float(vis))
                puntos.append([float(x), float(y)])

            if not valido:
                continue

            count_ok = sum(
                v >= VISIBILIDAD_MIN_BUSQUEDA for v in visibilidades
            )
            if count_ok < 3:
                continue

            puntos_arr = np.array(puntos, dtype=np.float32)

            # --- Validación geométrica ---
            if not self._area_es_coherente(puntos_arr):
                continue

            score = sum(visibilidades)
            if count_ok == 4:
                score += 1.0

            if score > mejor_score:
                mejor_score = score
                mejores_puntos = puntos_arr
                mejores_indices = indices

        return mejor_score, mejores_puntos, mejores_indices

    def _buscar_mejor_frame(self, frames):
        mejor_score = -1.0
        mejor_frame = None
        mejores_puntos = None
        mejores_indices = None

        total = len(frames)
        print(f"[AutoCalibrador] Buscando mejor frame en {total} frames "
              f"(paso {PASO_BUSQUEDA})...")

        frames_revisados = 0
        frames_con_deteccion = 0

        for frame_num in range(0, total, PASO_BUSQUEDA):
            frames_revisados += 1
            kpts = self._leer_keypoints(frames[frame_num])
            if kpts is None:
                continue

            score, puntos, indices = self._visibilidad_area(kpts)
            if puntos is None:
                continue

            frames_con_deteccion += 1

            if score > mejor_score:
                mejor_score = score
                mejor_frame = frame_num
                mejores_puntos = puntos
                mejores_indices = indices

        print(f"[AutoCalibrador] Frames revisados: {frames_revisados}, "
              f"con detección válida: {frames_con_deteccion}")

        if mejor_frame is None:
            print("[AutoCalibrador] No se encontró ningún frame con "
                  "visibilidad y geometría suficientes.")
            return None, None

        print(f"[AutoCalibrador] Mejor frame: {mejor_frame} "
              f"(score={mejor_score:.2f}, indices={mejores_indices})")
        print(f"[AutoCalibrador] Puntos: {mejores_puntos.tolist()}")

        self.mejor_frame = mejor_frame
        self.mejores_puntos = mejores_puntos

        return mejor_frame, mejores_puntos

    # ---------------------------------------------------------------
    # PROPAGACIÓN CON OPTICAL FLOW
    # ---------------------------------------------------------------
    def _guardar_homografia(self, frame_num, puntos_px):
        if puntos_px is None or len(puntos_px) != 4:
            return False

        puntos_px = np.asarray(puntos_px, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(puntos_px).all():
            return False

        H, _ = cv2.findHomography(puntos_px, self.field_m, cv2.RANSAC, 5.0)
        if H is None or not np.isfinite(H).all():
            return False
        if abs(np.linalg.det(H)) < 1e-12:
            return False

        self.homografias[frame_num] = H
        self.pixel_vertices[frame_num] = puntos_px.copy()
        return True

    def _propagar_adelante(self, frames, frame_inicial, puntos_iniciales):
        gray_prev = cv2.cvtColor(
            frames[frame_inicial], cv2.COLOR_BGR2GRAY
        )
        puntos = puntos_iniciales.copy()

        for frame_num in range(frame_inicial + 1, len(frames)):
            gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            nuevos, status, _ = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray,
                puntos.reshape(-1, 1, 2),
                None,
                **LK_PARAMS,
            )

            if nuevos is None or status is None or status.sum() < 4:
                gray_prev = gray
                continue

            puntos = nuevos.reshape(-1, 2).astype(np.float32)
            self._guardar_homografia(frame_num, puntos)
            gray_prev = gray

    def _propagar_atras(self, frames, frame_inicial, puntos_iniciales):
        gray_prev = cv2.cvtColor(
            frames[frame_inicial], cv2.COLOR_BGR2GRAY
        )
        puntos = puntos_iniciales.copy()

        for frame_num in range(frame_inicial - 1, -1, -1):
            gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            nuevos, status, _ = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray,
                puntos.reshape(-1, 1, 2),
                None,
                **LK_PARAMS,
            )

            if nuevos is None or status is None or status.sum() < 4:
                gray_prev = gray
                continue

            puntos = nuevos.reshape(-1, 2).astype(np.float32)
            self._guardar_homografia(frame_num, puntos)
            gray_prev = gray

    # ---------------------------------------------------------------
    # API PÚBLICA
    # ---------------------------------------------------------------
    def calibrar_frames(self, frames, cada_n=None):
        if not self.enabled:
            return 0

        if not frames:
            return 0

        frame_inicial, puntos_iniciales = self._buscar_mejor_frame(frames)
        if frame_inicial is None or puntos_iniciales is None:
            return 0

        if not self._guardar_homografia(frame_inicial, puntos_iniciales):
            print("[AutoCalibrador] No se pudo calcular homografía inicial.")
            return 0

        print("[AutoCalibrador] Propagando con optical flow...")
        self._propagar_adelante(frames, frame_inicial, puntos_iniciales)
        self._propagar_atras(frames, frame_inicial, puntos_iniciales)

        print(f"[AutoCalibrador] Homografías totales: "
              f"{len(self.homografias)}/{len(frames)}")

        return len(self.homografias)

    def _homografia_mas_cercana(self, frame_num):
        if not self.homografias:
            return None, None
        if frame_num in self.homografias:
            return (
                self.homografias[frame_num],
                self.pixel_vertices[frame_num],
            )
        frames_disp = sorted(self.homografias.keys())
        mas_cercano = min(frames_disp, key=lambda f: abs(f - frame_num))
        return (
            self.homografias[mas_cercano],
            self.pixel_vertices[mas_cercano],
        )


    def transform_point(self, point, frame_num=0):
        """Transforma un punto de imagen a metros usando la homografía."""

        if not self.enabled or point is None or not self.homografias:
            return None

        try:
            p = np.asarray(point, dtype=np.float32).reshape(-1)

            if p.size < 2:
                return None

            p = p[:2]

            if not np.isfinite(p).all():
                return None

            H, _ = self._homografia_mas_cercana(frame_num)

            if H is None or not np.isfinite(H).all():
                return None

            # No rechazar el balón por estar fuera del área penal.
            # La homografía se puede extrapolar al resto de la cancha.
            transformed = cv2.perspectiveTransform(
                p.reshape(1, 1, 2),
                H
            ).reshape(2)

            if not np.isfinite(transformed).all():
                return None

            return transformed.astype(float)

        except (ValueError, TypeError, IndexError, cv2.error):
            return None


    def add_transformed_position_to_tracks(self, tracks):
        if not self.enabled or not self.homografias:
            for _, object_tracks in tracks.items():
                for frame_tracks in object_tracks:
                    for item in frame_tracks.values():
                        item["position_transformed"] = None
            return

        for _, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for item in frame_tracks.values():
                    pos = item.get("position")
                    item["position_transformed"] = self.transform_point(
                        pos, frame_num=frame_num
                    )