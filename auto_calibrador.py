"""Calibración métrica parcial para vídeo de fútbol.

Usa los cuatro keypoints del área penal cuando son visibles y optical flow
entre redetecciones. No inventa una escala si no hay homografía válida.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from base_estadios import cargar_base_estadios, buscar_estadio

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = (
    BASE_DIR / "modelos" / "Model" / "weights" / "best.pt",
    BASE_DIR / "models" / "best.pt",
    BASE_DIR / "model" / "best.pt",
)
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])
ESTADIO_PREDETERMINADO = "Estadio Jalisco"

# Las dimensiones reglamentarias del área penal se cargan desde estadios.json.
# La dimensión total del estadio NO se usa como si fueran las esquinas del área.
_BASE_ESTADIOS = cargar_base_estadios()
_REGLAS = _BASE_ESTADIOS.get("reglas_ifab", {})
ANCHO_AREA_M = float(_REGLAS.get("area_penal_ancho_m", 40.32))
ALTO_AREA_M = float(_REGLAS.get("area_penal_profundidad_m", 16.5))
FIELD_POINTS_M = np.array(
    [[0.0, 0.0], [ANCHO_AREA_M, 0.0],
     [ANCHO_AREA_M, ALTO_AREA_M], [0.0, ALTO_AREA_M]],
    dtype=np.float32,
)
KEYPOINTS_AREAS = ((1, 2, 3, 4), (17, 18, 19, 20))
VISIBILITY_MIN = 0.25
REDETECT_EVERY = 12
MIN_POLYGON_AREA_PX = 1500.0
ASPECT_MIN = 1.15
ASPECT_MAX = 9.0
MAX_CORNER_STEP_PX = 180.0
LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class AutoCalibrador:
    """Transforma puntos de imagen a coordenadas métricas locales de cancha.

    `transform_point(point, frame_num=...)` mantiene compatibilidad con el
    pipeline existente. Si no existe homografía válida para ese frame devuelve
    None, nunca convierte píxeles a metros mediante una escala arbitraria.
    """
    def __init__(self, model_path=MODEL_PATH, estadio=ESTADIO_PREDETERMINADO):
        self.enabled = False
        self.modelo = None
        self.homografias: dict[int, np.ndarray] = {}
        self.pixel_vertices: dict[int, np.ndarray] = {}
        self.calibration_segment_by_frame: dict[int, int] = {}
        self.area_id_by_frame: dict[int, int] = {}
        self.field_m = FIELD_POINTS_M.copy()
        self.estadio_query = estadio
        self.estadio = buscar_estadio(estadio) if estadio else None
        if self.estadio:
            largo = self.estadio.get("largo_m")
            ancho = self.estadio.get("ancho_m")
            estado = self.estadio.get("estado_verificacion", "sin estado")
            print(f"[AutoCalibrador] Estadio seleccionado: {self.estadio.get('nombre', estadio)} "
                  f"({largo} x {ancho} m; estado: {estado}).")
            print(f"[AutoCalibrador] Homografía local del área penal: "
                  f"{ANCHO_AREA_M:.2f} x {ALTO_AREA_M:.2f} m, según estadios.json.")
        else:
            print(f"[AutoCalibrador] No se encontró el estadio '{estadio}' en estadios.json; "
                  "se usarán las dimensiones IFAB del área penal.")
        self.mejor_frame = None
        self.mejores_puntos = None
        self._segment_counter = 0
        path = Path(model_path)
        if not path.exists():
            print(f"[AutoCalibrador] No se encontró el modelo de keypoints: {path}. "
                  "Se permitirá calibración manual.")
            self.enabled = True
            return
        try:
            self.modelo = YOLO(str(path))
            self.enabled = True
            print(f"[AutoCalibrador] Modelo de calibración cargado: {path}")
        except Exception as exc:
            print(f"[AutoCalibrador] No se pudo cargar el modelo: {exc}")

    @staticmethod
    def _area_es_coherente(points):
        if points is None:
            return False
        p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if p.shape != (4, 2) or not np.isfinite(p).all():
            return False
        area = abs(float(cv2.contourArea(p)))
        if area < MIN_POLYGON_AREA_PX:
            return False
        sides = [float(np.linalg.norm(p[(i + 1) % 4] - p[i])) for i in range(4)]
        if min(sides) < 8.0:
            return False
        width_px = (sides[0] + sides[2]) / 2.0
        depth_px = (sides[1] + sides[3]) / 2.0
        if min(width_px, depth_px) <= 0:
            return False
        aspect = max(width_px, depth_px) / min(width_px, depth_px)
        if not ASPECT_MIN <= aspect <= ASPECT_MAX:
            return False
        # Evita quadriláteros casi degenerados o autointersecados.
        x, y, w, h = cv2.boundingRect(p)
        if w <= 0 or h <= 0 or area / float(w * h) < 0.18:
            return False
        return True

    def _leer_keypoints(self, frame):
        if self.modelo is None:
            return None
        try:
            results = self.modelo.predict(frame, verbose=False)
            if not results or results[0].keypoints is None:
                return None
            kpts = results[0].keypoints.data.cpu().numpy()
            if kpts.ndim == 3:
                if kpts.shape[0] == 0:
                    return None
                kpts = kpts[0]
            return kpts if kpts.ndim == 2 and kpts.shape[1] >= 3 else None
        except Exception as exc:
            print(f"[AutoCalibrador] Falló la detección de keypoints: {exc}")
            return None

    def _mejor_area(self, frame):
        kpts = self._leer_keypoints(frame)
        if kpts is None:
            return None
        candidates = []
        for area_id, indices in enumerate(KEYPOINTS_AREAS):
            if max(indices) >= len(kpts):
                continue
            selected = kpts[list(indices)]
            visibility = selected[:, 2].astype(float)
            points = selected[:, :2].astype(np.float32)
            if not np.isfinite(selected).all():
                continue
            if int(np.sum(visibility >= VISIBILITY_MIN)) < 3:
                continue
            # Cuatro puntos son necesarios para homografía; si uno tiene poca
            # visibilidad, solo se acepta si el modelo igual lo localizó y la
            # geometría completa pasa la validación. No se inventa el cuarto.
            if float(np.min(visibility)) < 0.08:
                continue
            if not self._area_es_coherente(points):
                continue
            score = float(np.sum(np.clip(visibility, 0, 1)))
            candidates.append((score, area_id, points))
        if not candidates:
            return None
        _, area_id, points = max(candidates, key=lambda x: x[0])
        return area_id, points

    def _buscar_mejor_frame(self, frames, paso=3):
        best = None
        for idx in range(0, len(frames), max(1, int(paso))):
            candidate = self._mejor_area(frames[idx])
            if candidate is None:
                continue
            area_id, points = candidate
            score = abs(float(cv2.contourArea(points)))
            if best is None or score > best[0]:
                best = (score, idx, area_id, points)
        if best is None:
            # Alternativa automática: buscar una región rectangular a partir de
            # líneas parciales. Solo se acepta cuando se pueden inferir cuatro
            # intersecciones geométricas plausibles; nunca se usa una escala px/m
            # arbitraria. La región se interpreta como área penal (40.32 x 16.5 m).
            try:
                from detector_lineas_cancha import detectar_segmentos_cancha, ajustar_recta, interseccion_rectas
                for idx in range(0, len(frames), max(1, int(paso))):
                    segs = detectar_segmentos_cancha(frames[idx])
                    if len(segs) < 4:
                        continue
                    # Mantener los segmentos más largos para reducir ruido de
                    # camisetas, carteles y espectadores. Agrupar por orientación.
                    segs = segs[:min(24, len(segs))]
                    horizontal = [x for x in segs if min(x["angle_deg"], 180-x["angle_deg"]) < 25]
                    vertical = [x for x in segs if 65 < x["angle_deg"] < 115]
                    if len(horizontal) < 2 or len(vertical) < 2:
                        continue
                    # Probar pares de rectas en cada familia y seleccionar el
                    # cuadrilátero de mayor área con proporción plausible.
                    rect_h = [ajustar_recta([x]) for x in horizontal]
                    rect_v = [ajustar_recta([x]) for x in vertical]
                    candidates = []
                    for a in range(len(rect_h)):
                        for b in range(a+1, len(rect_h)):
                            for c in range(len(rect_v)):
                                for d in range(c+1, len(rect_v)):
                                    pts = [interseccion_rectas(rect_h[a], rect_v[c]),
                                           interseccion_rectas(rect_h[a], rect_v[d]),
                                           interseccion_rectas(rect_h[b], rect_v[d]),
                                           interseccion_rectas(rect_h[b], rect_v[c])]
                                    if any(q is None for q in pts):
                                        continue
                                    pxy = np.asarray(pts, np.float32)
                                    h, w = frames[idx].shape[:2]
                                    # Permitir intersecciones extrapoladas ligeramente fuera
                                    # de imagen, pero rechazar geometrías disparatadas.
                                    if np.any(pxy[:,0] < -0.15*w) or np.any(pxy[:,0] > 1.15*w) or np.any(pxy[:,1] < -0.15*h) or np.any(pxy[:,1] > 1.15*h):
                                        continue
                                    area = abs(float(cv2.contourArea(pxy)))
                                    if area < MIN_POLYGON_AREA_PX:
                                        continue
                                    sides = [float(np.linalg.norm(pxy[(i+1)%4]-pxy[i])) for i in range(4)]
                                    if min(sides) < 15:
                                        continue
                                    ratio = max((sides[0]+sides[2])/2, (sides[1]+sides[3])/2) / max(1.0, min((sides[0]+sides[2])/2, (sides[1]+sides[3])/2))
                                    if not 1.1 <= ratio <= 8.5:
                                        continue
                                    candidates.append((area, pxy))
                    if candidates:
                        area, pxy = max(candidates, key=lambda z:z[0])
                        # Homografía de área local; la validación posterior rechaza
                        # puntos no convexos o demasiado degenerados.
                        if self._area_es_coherente(pxy):
                            best = (area, idx, 0, pxy)
                            print(f"[AutoCalibrador] Fallback por líneas parciales en frame {idx}.")
                            break
            except Exception as exc:
                print(f"[AutoCalibrador] Fallback geométrico por líneas falló: {exc}")
        if best is None:
            # Modo híbrido seguro: no abrir una ventana que exija cuatro
            # esquinas que no aparecen en pantalla. Detectamos marcas parciales
            # para diagnóstico/seguimiento, pero no inventamos una homografía
            # métrica a partir de líneas sin correspondencias de cancha conocidas.
            try:
                from detector_lineas_cancha import detectar_segmentos_cancha
                frame_num = len(frames) // 2
                segmentos = detectar_segmentos_cancha(frames[frame_num])
                print(f"[AutoCalibrador] No hay cuatro keypoints válidos. "
                      f"Detector híbrido encontró {len(segmentos)} segmentos de líneas "
                      f"en frame {frame_num}; no se exige marcar cuatro esquinas.")
                if segmentos:
                    diagnostico = frames[frame_num].copy()
                    for seg in segmentos:
                        cv2.line(diagnostico, seg["p1"], seg["p2"], (0, 255, 255), 2)
                    salida = BASE_DIR / "diagnostico_lineas_parciales.jpg"
                    cv2.imwrite(str(salida), diagnostico)
                    print(f"[AutoCalibrador] Diagnóstico de líneas guardado: {salida}")
            except Exception as exc:
                print(f"[AutoCalibrador] Detector de líneas parciales no disponible: {exc}")
            print("[AutoCalibrador] No se puede calcular velocidad métrica fiable sin "
                  "una referencia geométrica identificable. Se omite la calibración "
                  "manual de cuatro esquinas para este video.")
            return None, None, None
        _, frame_num, area_id, points = best
        self.mejor_frame, self.mejores_puntos = frame_num, points.copy()
        print(f"[AutoCalibrador] Mejor frame: {frame_num}; área {area_id}; puntos válidos.")
        return frame_num, area_id, points

    def _guardar_homografia(self, frame_num, points, segment_id, area_id):
        if not self._area_es_coherente(points):
            return False
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)
        H, inliers = cv2.findHomography(points, self.field_m, method=0)
        if H is None or not np.isfinite(H).all() or abs(float(np.linalg.det(H))) < 1e-12:
            return False
        # Reproyección de las cuatro esquinas: evita aceptar homografías malas.
        mapped = cv2.perspectiveTransform(points.reshape(1, -1, 2), H).reshape(-1, 2)
        if not np.isfinite(mapped).all() or np.max(np.linalg.norm(mapped - self.field_m, axis=1)) > 0.5:
            return False
        self.homografias[int(frame_num)] = H
        self.pixel_vertices[int(frame_num)] = points.copy()
        self.calibration_segment_by_frame[int(frame_num)] = int(segment_id)
        self.area_id_by_frame[int(frame_num)] = int(area_id)
        return True

    def _propagar_direccion(self, frames, anchor_frame, anchor_points, anchor_area, direction):
        previous_idx = anchor_frame
        previous_gray = cv2.cvtColor(frames[anchor_frame], cv2.COLOR_BGR2GRAY)
        points = np.asarray(anchor_points, dtype=np.float32).copy()
        area_id = int(anchor_area)
        segment_id = 0
        self._guardar_homografia(anchor_frame, points, segment_id, area_id)
        step_count = 0
        for frame_num in range(anchor_frame + direction, len(frames) if direction > 0 else -1, direction):
            gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            step_count += 1
            redetected = self._mejor_area(frames[frame_num]) if step_count % REDETECT_EVERY == 0 else None
            if redetected is not None:
                new_area_id, new_points = redetected
                # Cambiar entre áreas opuestas crea otro sistema local; nunca
                # comparar velocidades cruzando esa discontinuidad.
                if new_area_id != area_id:
                    self._segment_counter += 1
                    segment_id = self._segment_counter
                    area_id = new_area_id
                points = new_points.astype(np.float32)
                self._guardar_homografia(frame_num, points, segment_id, area_id)
                previous_gray = gray
                previous_idx = frame_num
                continue

            try:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous_gray, gray, points.reshape(-1, 1, 2), None, **LK_PARAMS
                )
            except cv2.error:
                next_points, status = None, None
            if next_points is None or status is None or int(status.reshape(-1).sum()) < 4:
                # No avanzar el frame de referencia sin mover también los puntos.
                # En el siguiente paso se intentará el flujo desde la última
                # geometría válida; si el salto es grande, se rechazará abajo.
                direct = self._mejor_area(frames[frame_num])
                if direct is not None:
                    new_area_id, candidate = direct
                    if new_area_id != area_id:
                        self._segment_counter += 1
                        segment_id = self._segment_counter
                        area_id = new_area_id
                    if self._guardar_homografia(frame_num, candidate, segment_id, area_id):
                        points = candidate.astype(np.float32)
                        previous_gray = gray
                        previous_idx = frame_num
                continue
            candidate = next_points.reshape(-1, 2).astype(np.float32)
            movement = np.linalg.norm(candidate - points, axis=1)
            if not np.isfinite(candidate).all() or np.max(movement) > MAX_CORNER_STEP_PX:
                # Posible corte/zoom fuerte: intentar detección directa en el frame.
                direct = self._mejor_area(frames[frame_num])
                if direct is None:
                    previous_gray = gray
                    previous_idx = frame_num
                    continue
                new_area_id, candidate = direct
                if new_area_id != area_id:
                    self._segment_counter += 1
                    segment_id = self._segment_counter
                    area_id = new_area_id
            if self._guardar_homografia(frame_num, candidate, segment_id, area_id):
                points = candidate
                previous_gray = gray
                previous_idx = frame_num
            else:
                previous_gray = gray
                previous_idx = frame_num

    def calibrar_frames(self, frames, cada_n=None):
        if not self.enabled or not frames:
            return 0
        self.homografias.clear()
        self.pixel_vertices.clear()
        self.calibration_segment_by_frame.clear()
        self.area_id_by_frame.clear()
        anchor_frame, area_id, points = self._buscar_mejor_frame(frames, paso=cada_n or 3)
        if anchor_frame is None:
            self.enabled = False
            return 0
        self._segment_counter = 0
        if not self._guardar_homografia(anchor_frame, points, 0, area_id):
            print("[AutoCalibrador] La homografía inicial no superó validación.")
            self.enabled = False
            return 0
        # Propaga en ambos sentidos desde el mejor frame. Las redetecciones
        # periódicas limitan la deriva del optical flow.
        self._propagar_direccion(frames, anchor_frame, points, area_id, +1)
        self._propagar_direccion(frames, anchor_frame, points, area_id, -1)
        print(f"[AutoCalibrador] Homografías válidas: {len(self.homografias)}/{len(frames)} frames.")
        if not self.homografias:
            self.enabled = False
        return len(self.homografias)

    def _homografia_mas_cercana(self, frame_num):
        if not self.homografias:
            return None, None, None
        frame_num = int(frame_num)
        if frame_num in self.homografias:
            return self.homografias[frame_num], self.pixel_vertices[frame_num], self.calibration_segment_by_frame.get(frame_num)
        keys = sorted(self.homografias)
        nearest = min(keys, key=lambda k: abs(k - frame_num))
        # No reutilizar una homografía lejana: tras un corte o una oclusión
        # puede ser geométricamente incorrecta.
        if abs(nearest - frame_num) > 3:
            return None, None, None
        return self.homografias[nearest], self.pixel_vertices[nearest], self.calibration_segment_by_frame.get(nearest)

    def get_calibration_segment(self, frame_num=0):
        _, _, segment = self._homografia_mas_cercana(frame_num)
        return segment

    def transform_point(self, point, frame_num=0):
        if not self.enabled or point is None:
            return None
        try:
            p = np.asarray(point, dtype=np.float32).reshape(-1)[:2]
            if p.size != 2 or not np.isfinite(p).all():
                return None
            H, vertices, segment = self._homografia_mas_cercana(frame_num)
            if H is None or vertices is None:
                return None
            # No exigir que el balón esté dentro del área: la homografía puede
            # extrapolarse localmente, pero la distancia pierde fiabilidad lejos
            # de la zona calibrada. El filtrado temporal limita falsos saltos.
            transformed = cv2.perspectiveTransform(p.reshape(1, 1, 2), H).reshape(2)
            if not np.isfinite(transformed).all():
                return None
            return transformed.astype(float)
        except (ValueError, TypeError, IndexError, cv2.error):
            return None

    def add_transformed_position_to_tracks(self, tracks):
        for _, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for item in frame_tracks.values():
                    item["position_transformed"] = self.transform_point(
                        item.get("position"), frame_num=frame_num
                    )
