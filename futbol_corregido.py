# ============================================================
# FUTBOL_CORREGIDO.PY - VERSIÓN FINAL CON POSESIÓN Y GOLEADOR
# ============================================================

import os
import sys
import json
import math
import pickle
from pathlib import Path
from collections import Counter, defaultdict, deque

try:
    from view_transformer_tramos import InterpolatedViewTransformer
except (ImportError, AttributeError) as exc:
    InterpolatedViewTransformer = None
    print(
        "[Calibración] Módulo de tramos no disponible o sin "
        "InterpolatedViewTransformer; se usará el calibrador automático. "
        f"Detalle: {exc}"
    )

from auto_calibrador import AutoCalibrador

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from ultralytics import YOLO
import supervision as sv

from defensee import (
    detectar_alertas_defensivas,
    resumir_secuencias_ataque,
    distancia_jugador_en_intervalo,
    metricas_jugador_con_sin_pelota,
    distancia_autor_gol,
)

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_VIDEO_PATH = Path(
    r"C:\Users\Matias\Downloads\messii gol.mp4"
)

VIDEO_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH

CLIP_NAME = VIDEO_PATH.stem

MODEL_PATH_CANDIDATES = [
    BASE_DIR / "models" / "best.pt",
    BASE_DIR / "modelos" / "Model" / "weights" / "best.pt",
]
MODEL_PATH = next((p for p in MODEL_PATH_CANDIDATES if p.exists()), MODEL_PATH_CANDIDATES[0])
OUTPUT_DIR = BASE_DIR / "output_videos"
STUBS_DIR = BASE_DIR / "stubs"

TRACKS_STUB = STUBS_DIR / f"{CLIP_NAME}_track_stubs.pkl"
CAMERA_STUB = STUBS_DIR / f"{CLIP_NAME}_camera_movement_stub.pkl"

TACTICAL_VIDEO_PATH = OUTPUT_DIR / f"analisis_tactico_{CLIP_NAME}.avi"
REPORT_PATH = OUTPUT_DIR / f"tactical_report_{CLIP_NAME}.csv"
SUMMARY_PATH = OUTPUT_DIR / f"tactical_summary_{CLIP_NAME}.csv"
MAP_PATH = OUTPUT_DIR / f"tactical_map_{CLIP_NAME}.png"
DEBUG_TRACKING_PATH = OUTPUT_DIR / f"debug_tracking_{CLIP_NAME}.png"
CALIBRATION_PATH = BASE_DIR / "calibracion_cancha.json"

# Detección de ataques y desenlaces automática; no se cargan minutos de gol manualmente.
GOAL_TIMES_SECONDS = []

MAP_SAMPLE_EVERY = 5
DETECTION_CONFIDENCE = 0.10
DETECTION_BATCH_SIZE = 20

TEAM_ARGENTINA = 1
TEAM_ENGLAND = 2
TEAM_NEUTRAL = 0

TEAM_COLORS = {
    TEAM_ARGENTINA: (255, 120, 0),
    TEAM_ENGLAND: (0, 80, 255),
    TEAM_NEUTRAL: (180, 180, 180),
}

GOALKEEPER_COLOR = (0, 220, 255)
REFEREE_COLOR = (0, 255, 0)

WHITE = (255, 255, 255)
GREEN = (70, 220, 90)
YELLOW = (0, 220, 255)
RED = (0, 0, 255)
DARK = (25, 25, 25)

EXPECTED_TEAM_PLAYERS = 11
MIN_TEAM_VOTES = 3
MIN_TEAM_VOTE_RATIO = 0.60

MIN_PLAYER_HEIGHT_PX = 18
LINE_GAP_FACTOR = 2.2
MAX_FORMATION_LINES = 4

ALTURA_PROMEDIO_JUGADOR_M = 1.75
ANCHO_ARCO_M = 7.32
ALTO_ARCO_M = 2.44

# ============================================================
# UTILIDADES
# ============================================================

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_bbox_width(bbox):
    return float(bbox[2] - bbox[0])

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))

def measure_distance(p1, p2):
    return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)))

def bbox_is_valid(bbox):
    if bbox is None or len(bbox) < 4:
        return False
    values = np.asarray(bbox[:4], dtype=float)
    if not np.all(np.isfinite(values)):
        return False
    x1, y1, x2, y2 = values
    return x2 > x1 and y2 > y1

def player_bbox_size(player):
    bbox = player.get("bbox")
    if not bbox_is_valid(bbox):
        return 0.0, 0.0
    return (float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))

def is_valid_player_detection(player):
    if not bbox_is_valid(player.get("bbox")):
        return False
    width, height = player_bbox_size(player)
    return width > 0 and height >= MIN_PLAYER_HEIGHT_PX

def is_neutral_team(team):
    return team not in (TEAM_ARGENTINA, TEAM_ENGLAND)

def safe_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0:
        return 25.0
    return float(fps)

# ============================================================
# CALIBRACIÓN POR REFERENCIA HUMANA Y ARCO
# ============================================================

def calcular_escala_local_por_jugador(player):
    _, altura_px = player_bbox_size(player)
    if altura_px <= 0:
        return None
    return ALTURA_PROMEDIO_JUGADOR_M / altura_px

def estimar_escala_balon_por_jugador_cercano(players, ball_pos):
    if ball_pos is None:
        return None
    mejor_distancia = float('inf')
    mejor_escala = None
    for player in players.values():
        if player.get("role") == "referee":
            continue
        pos = get_player_position(player)
        if pos is None:
            continue
        dist = np.linalg.norm(pos - ball_pos)
        if dist < mejor_distancia:
            escala = calcular_escala_local_por_jugador(player)
            if escala is not None:
                mejor_distancia = dist
                mejor_escala = escala
    return mejor_escala

def calcular_escala_porteria(bbox_arco_px):
    if bbox_arco_px is None or len(bbox_arco_px) < 4:
        return None
    ancho_px = bbox_arco_px[2] - bbox_arco_px[0]
    alto_px = bbox_arco_px[3] - bbox_arco_px[1]
    if ancho_px <= 0 or alto_px <= 0:
        return None
    escala_x = ANCHO_ARCO_M / ancho_px
    escala_y = ALTO_ARCO_M / alto_px
    return (escala_x + escala_y) / 2.0

def calcular_distancia_recorrida(player_actual, player_anterior, escala_m_px):
    if player_actual is None or player_anterior is None or escala_m_px is None:
        return 0.0
    pos_actual = get_player_position(player_actual)
    pos_anterior = get_player_position(player_anterior)
    if pos_actual is None or pos_anterior is None:
        return 0.0
    dist_px = np.linalg.norm(pos_actual - pos_anterior)
    return dist_px * escala_m_px

# ============================================================
# VIDEO
# ============================================================

def read_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"El video no contiene fotogramas: {video_path}")
    return frames

# ============================================================
# TRACKER
# ============================================================
class Tracker:
    """Detector + ByteTrack con recuperación conservadora de IDs por apariencia.

    ByteTrack puede asignar un ID nuevo después de una oclusión. Este wrapper
    mantiene un ID estable cuando reaparece un jugador con apariencia visual
    muy similar y un intervalo de ausencia corto. No intenta adivinar si la
    apariencia no ofrece evidencia suficiente.
    """
    REID_MAX_GAP_FRAMES = 75
    REID_MIN_CORRELATION = 0.88

    def __init__(self, model_path):
        self.model = YOLO(str(model_path))
        self.tracker = sv.ByteTrack()
        self._raw_to_stable = {}
        self._appearance = {}
        self._last_seen = {}
        self._last_role = {}
        self._last_bbox = {}
        self._next_stable_id = 1

    def detect_frames(self, frames):
        detections = []
        for start in range(0, len(frames), DETECTION_BATCH_SIZE):
            batch = frames[start:start + DETECTION_BATCH_SIZE]
            results = self.model.predict(batch, conf=DETECTION_CONFIDENCE, verbose=False)
            detections.extend(results)
        return detections

    @staticmethod
    def _appearance_descriptor(frame, bbox):
        """Histograma HSV del torso/parte central; no requiere modelos extra."""
        if frame is None or not bbox_is_valid(bbox):
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox[:4])
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 - x1 < 8 or y2 - y1 < 12:
            return None
        # Se reduce el fondo y se prioriza la camiseta, útil para reidentificación.
        crop = frame[y1 + int((y2-y1)*0.12):y1 + int((y2-y1)*0.78),
                     x1 + int((x2-x1)*0.15):x1 + int((x2-x1)*0.85)]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist

    @staticmethod
    def _appearance_similarity(first, second):
        if first is None or second is None:
            return -1.0
        try:
            value = cv2.compareHist(first.astype(np.float32), second.astype(np.float32), cv2.HISTCMP_CORREL)
            return float(value) if np.isfinite(value) else -1.0
        except cv2.error:
            return -1.0

    def _stable_id_for(self, raw_id, role, descriptor, bbox, frame_num, used_stable_ids):
        if raw_id in self._raw_to_stable:
            stable_id = self._raw_to_stable[raw_id]
            # Si el ID crudo fue reutilizado mientras el ID estable ya está
            # ocupado en este frame, se trata como una identidad nueva.
            if stable_id not in used_stable_ids:
                return stable_id
            self._raw_to_stable.pop(raw_id, None)

        best_id, best_score = None, self.REID_MIN_CORRELATION
        current_height = max(1.0, float(bbox[3]) - float(bbox[1]))
        for candidate_id, old_hist in self._appearance.items():
            if candidate_id in used_stable_ids:
                continue
            if self._last_role.get(candidate_id) != role:
                continue
            gap = frame_num - self._last_seen.get(candidate_id, -10**9)
            if gap < 1 or gap > self.REID_MAX_GAP_FRAMES:
                continue
            old_bbox = self._last_bbox.get(candidate_id)
            if bbox_is_valid(old_bbox):
                old_height = max(1.0, float(old_bbox[3]) - float(old_bbox[1]))
                ratio = current_height / old_height
                # Evita asociar jugadores muy distintos en tamaño por perspectiva.
                if ratio < 0.58 or ratio > 1.72:
                    continue
            score = self._appearance_similarity(descriptor, old_hist)
            if score > best_score:
                best_id, best_score = candidate_id, score

        if best_id is None:
            best_id = self._next_stable_id
            self._next_stable_id += 1
        self._raw_to_stable[raw_id] = best_id
        return best_id

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            try:
                with open(stub_path, "rb") as file:
                    cached = pickle.load(file)
                if (len(cached.get("players", [])) == len(frames)
                        and len(cached.get("ball", [])) == len(frames)):
                    print("[Tracker] Stub válido.")
                    return cached
            except Exception as exc:
                print(f"[Tracker] No se pudo leer el stub: {exc}")

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}
        for frame_num, detection in enumerate(detections):
            frame = frames[frame_num]
            names = {int(class_id): str(name).strip().lower() for class_id, name in detection.names.items()}
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            dets = sv.Detections.from_ultralytics(detection)
            tracked_dets = self.tracker.update_with_detections(dets)
            used_stable_ids = set()

            for item in tracked_dets:
                bbox = item[0].tolist()
                class_id_value = item[3]
                raw_id_value = item[4]
                if class_id_value is None or raw_id_value is None or not bbox_is_valid(bbox):
                    continue
                class_name = names.get(int(class_id_value), "")
                if class_name == "player":
                    role = "field_player"
                elif class_name == "goalkeeper":
                    role = "goalkeeper"
                elif class_name == "referee":
                    role = "referee"
                else:
                    continue

                raw_id = int(raw_id_value)
                descriptor = self._appearance_descriptor(frame, bbox)
                stable_id = self._stable_id_for(
                    raw_id, role, descriptor, bbox, frame_num, used_stable_ids
                )
                used_stable_ids.add(stable_id)
                info = {"bbox": bbox, "role": role, "class_name": class_name,
                        "raw_track_id": raw_id}
                tracks["players"][frame_num][stable_id] = info
                if role == "referee":
                    tracks["referees"][frame_num][stable_id] = info.copy()

                # Actualización temporal del descriptor para adaptarse a cambios
                # moderados de luz sin sustituirlo bruscamente.
                previous_hist = self._appearance.get(stable_id)
                if descriptor is not None:
                    if previous_hist is not None:
                        descriptor = cv2.addWeighted(previous_hist.astype(np.float32), 0.65,
                                                     descriptor.astype(np.float32), 0.35, 0)
                        cv2.normalize(descriptor, descriptor, alpha=1.0, norm_type=cv2.NORM_L1)
                    self._appearance[stable_id] = descriptor
                self._last_seen[stable_id] = frame_num
                self._last_role[stable_id] = role
                self._last_bbox[stable_id] = bbox

            ball_candidates = []
            for item in dets:
                bbox = item[0].tolist()
                class_id_value = item[3]
                confidence = item[2]
                if class_id_value is None:
                    continue
                class_name = names.get(int(class_id_value), "")
                if class_name != "ball" or not bbox_is_valid(bbox):
                    continue
                confidence = float(confidence) if confidence is not None else 0.0
                ball_candidates.append((confidence, bbox))
            if ball_candidates:
                confidence, bbox = max(ball_candidates, key=lambda value: value[0])
                tracks["ball"][frame_num][1] = {
                    "bbox": bbox, "class_name": "ball", "confidence": confidence,
                }

        if stub_path:
            stub_dir = os.path.dirname(str(stub_path))
            if stub_dir:
                os.makedirs(stub_dir, exist_ok=True)
            with open(stub_path, "wb") as file:
                pickle.dump(tracks, file)
        return tracks

    def add_position_to_tracks(self, tracks):
        for category, category_tracks in tracks.items():
            if not isinstance(category_tracks, list):
                continue
            for frame_tracks in category_tracks:
                for info in frame_tracks.values():
                    bbox = info.get("bbox")
                    if not bbox_is_valid(bbox):
                        continue
                    position = get_center_of_bbox(bbox) if category == "ball" else get_foot_position(bbox)
                    info["position"] = tuple(position)

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        if not bbox_is_valid(bbox):
            return frame
        x_center, _ = get_center_of_bbox(bbox)
        y2 = int(bbox[3])
        width = max(8, int(get_bbox_width(bbox)))
        cv2.ellipse(frame, (x_center, y2),
                    (max(8, width // 2), max(4, int(width * 0.18))),
                    0, 0, 360, color, 2)
        if track_id is not None:
            cv2.putText(frame, str(track_id), (x_center - 10, y2 + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return frame

    def draw_traingle(self, frame, bbox, color):
        if not bbox_is_valid(bbox):
            return frame
        x, y = get_center_of_bbox(bbox)
        y = int(bbox[1])
        points = np.array([[x, y], [x - 8, y - 14], [x + 8, y - 14]], dtype=np.int32)
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 1)
        return frame

# ============================================================
# ROLES ESTABLES Y EQUIPOS
# ============================================================
def stabilize_track_roles(tracks):
    role_votes = defaultdict(Counter)
    for frame_players in tracks["players"]:
        for track_id, player in frame_players.items():
            class_name = player.get("class_name", "").lower()
            if class_name == "referee":
                role_votes[track_id]["referee"] += 1
            elif class_name == "goalkeeper":
                role_votes[track_id]["goalkeeper"] += 1
            elif class_name == "player":
                role_votes[track_id]["field_player"] += 1
    stable_roles = {}
    for track_id, votes in role_votes.items():
        if not votes:
            continue
        role, count = votes.most_common(1)[0]
        total = sum(votes.values())
        if total >= 3 and count / total < 0.50:
            role = "field_player"
        stable_roles[track_id] = role
    for frame_players in tracks["players"]:
        for track_id, player in frame_players.items():
            role = stable_roles.get(track_id, player.get("role", "field_player"))
            player["role"] = role
            if role == "referee":
                player["team"] = TEAM_NEUTRAL
                player["team_color"] = REFEREE_COLOR
    return stable_roles

class TeamAssigner:
    def __init__(self):
        self.kmeans = None
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_history = {}
        self.history_size = 9

    def get_player_color(self, frame, bbox):
        if frame is None or not bbox_is_valid(bbox):
            return None
        height_frame, width_frame = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox[:4])
        x1 = max(0, min(x1, width_frame - 1))
        x2 = max(0, min(x2, width_frame))
        y1 = max(0, min(y1, height_frame - 1))
        y2 = max(0, min(y2, height_frame))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        height, width = crop.shape[:2]
        if height < 20 or width < 10:
            return None
        torso = crop[int(height * 0.18):int(height * 0.52), int(width * 0.20):int(width * 0.80)]
        if torso.size == 0:
            return None
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        pixels = pixels[pixels[:, 2] > 35]
        if len(pixels) < 10:
            return None
        hue = float(np.median(pixels[:, 0])) / 180.0
        saturation = float(np.median(pixels[:, 1])) / 255.0
        value = float(np.median(pixels[:, 2])) / 255.0
        return np.array([hue * saturation, saturation, value], dtype=np.float32)

    def assign_team_color_multi_frame(self, frames, tracks, max_frames=60):
        total = min(len(frames), len(tracks["players"]))
        if total == 0:
            print("[Equipos] No hay fotogramas.")
            return False
        sample_indices = np.linspace(0, total - 1, num=min(max_frames, total), dtype=int)
        colors = []
        for frame_num in sample_indices:
            frame = frames[frame_num]
            for player in tracks["players"][frame_num].values():
                role = player.get("role", "field_player")
                if role in ("goalkeeper", "referee", "neutral"):
                    continue
                color = self.get_player_color(frame, player.get("bbox"))
                if color is not None:
                    colors.append(color)
        if len(colors) < 8:
            print("[Equipos] No hay suficientes muestras de color.")
            return False
        colors = np.asarray(colors, dtype=np.float32)
        if len(np.unique(np.round(colors, 3), axis=0)) < 2:
            print("[Equipos] Los colores detectados son demasiado similares.")
            return False
        model = KMeans(n_clusters=2, n_init=15, random_state=42)
        labels = model.fit_predict(colors)
        if len(np.unique(labels)) != 2:
            print("[Equipos] KMeans no pudo separar dos grupos.")
            return False
        self.kmeans = model
        self.team_colors = {1: model.cluster_centers_[0].copy(), 2: model.cluster_centers_[1].copy()}
        self.player_team_dict.clear()
        self.player_team_history.clear()
        print(f"[Equipos] KMeans entrenado con {len(colors)} muestras.")
        return True

    def get_player_team(self, frame, player_bbox, player_id, role="field_player"):
        if role in ("goalkeeper", "referee", "neutral"):
            return TEAM_NEUTRAL
        if self.kmeans is None:
            return TEAM_NEUTRAL
        color = self.get_player_color(frame, player_bbox)
        if color is None:
            return self.player_team_dict.get(player_id, TEAM_NEUTRAL)
        prediction = int(self.kmeans.predict(color.reshape(1, -1))[0]) + 1
        if player_id not in self.player_team_history:
            self.player_team_history[player_id] = deque(maxlen=self.history_size)
        history = self.player_team_history[player_id]
        history.append(prediction)
        counts = Counter(history)
        team, votes = counts.most_common(1)[0]
        if votes / len(history) < 0.60:
            return self.player_team_dict.get(player_id, TEAM_NEUTRAL)
        self.player_team_dict[player_id] = team
        return team

def collect_team_votes(tracks, team_assigner, frames):
    votes = defaultdict(list)
    total = min(len(frames), len(tracks["players"]))
    for frame_num in range(total):
        for track_id, player in tracks["players"][frame_num].items():
            if not is_valid_player_detection(player):
                continue
            role = player.get("role", "field_player")
            if role in ("goalkeeper", "referee", "neutral"):
                votes[track_id].append(TEAM_NEUTRAL)
                continue
            team = team_assigner.get_player_team(frames[frame_num], player["bbox"], track_id, role)
            votes[track_id].append(team)
    return votes

def resolve_track_teams(team_votes):
    resolved = {}
    for track_id, votes in team_votes.items():
        valid = [team for team in votes if team in (TEAM_ARGENTINA, TEAM_ENGLAND)]
        if len(valid) < MIN_TEAM_VOTES:
            resolved[track_id] = TEAM_NEUTRAL
            continue
        counts = Counter(valid)
        team, count = counts.most_common(1)[0]
        if count / len(valid) < MIN_TEAM_VOTE_RATIO:
            resolved[track_id] = TEAM_NEUTRAL
        else:
            resolved[track_id] = int(team)
    return resolved

def apply_resolved_teams(tracks, resolved):
    for frame_players in tracks["players"]:
        for track_id, player in frame_players.items():
            role = player.get("role", "field_player")
            if role == "referee":
                player["team"] = TEAM_NEUTRAL
                player["team_color"] = REFEREE_COLOR
                continue
            if role == "goalkeeper":
                player["team"] = resolved.get(track_id, TEAM_NEUTRAL)
                player["team_color"] = GOALKEEPER_COLOR
                continue
            team = resolved.get(track_id, TEAM_NEUTRAL)
            player["team"] = int(team)
            player["team_color"] = TEAM_COLORS.get(team, TEAM_COLORS[TEAM_NEUTRAL])

# ============================================================
# POSICIONES
# ============================================================
def get_player_position(player):
    bbox = player.get("bbox")
    if not bbox_is_valid(bbox):
        return None
    x1, y1, x2, y2 = map(float, bbox[:4])
    return np.array([(x1 + x2) / 2.0, y2], dtype=float)

def get_ball_position(ball):
    bbox = ball.get("bbox")
    if bbox_is_valid(bbox):
        x1, y1, x2, y2 = map(float, bbox[:4])
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
    position = ball.get("position")
    if position is not None and len(position) >= 2:
        point = np.asarray(position[:2], dtype=float)
        if np.all(np.isfinite(point)):
            return point
    return None

def tactical_position(player):
    for key in ("position_adjusted", "position"):
        value = player.get(key)
        if value is None or len(value) < 2:
            continue
        point = np.asarray(value[:2], dtype=float)
        if np.all(np.isfinite(point)):
            return point
    return None

# ============================================================
# MOVIMIENTO DE CÁMARA
# ============================================================
class CameraMovementEstimator:
    def __init__(self, frame):
        self.lk_params = {
            "winSize": (21, 21), "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            try:
                with open(stub_path, "rb") as file:
                    cached = pickle.load(file)
                if len(cached) == len(frames):
                    return cached
            except Exception as exc:
                print(f"[Cámara] No se pudo leer stub: {exc}")
        if not frames:
            return []
        movements = [(0.0, 0.0)]
        previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        previous_points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=7)
        for frame_num in range(1, len(frames)):
            gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            dx, dy = 0.0, 0.0
            if previous_points is not None and len(previous_points) >= 5:
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, previous_points, None, **self.lk_params)
                if new_points is not None and status is not None:
                    mask = status.reshape(-1) == 1
                    old_good = previous_points.reshape(-1, 2)[mask]
                    new_good = new_points.reshape(-1, 2)[mask]
                    if len(old_good) >= 5:
                        displacement = new_good - old_good
                        dx, dy = np.median(displacement, axis=0).tolist()
                    previous_points = new_points[status.reshape(-1) == 1]
            movements.append((float(dx), float(dy)))
            if previous_points is None or len(previous_points) < 20:
                previous_points = cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=7)
            previous_gray = gray
        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as file:
                pickle.dump(movements, file)
        return movements

    def add_adjust_positions_to_tracks(self, tracks, movements):
        if not movements:
            return
        movements = np.asarray(movements, dtype=float)
        cumulative = np.cumsum(movements, axis=0)
        for category, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                if frame_num >= len(cumulative):
                    continue
                dx, dy = cumulative[frame_num]
                for item in frame_tracks.values():
                    position = item.get("position")
                    if position is None:
                        continue
                    item["position_adjusted"] = (float(position[0] - dx), float(position[1] - dy))

# ============================================================
# CALIBRACIÓN DE CANCHA (VIEW TRANSFORMER)
# ============================================================
class ViewTransformer:
    def __init__(self, pixel_vertices=None, target_vertices=None):
        self.enabled = False
        self.pixel_vertices = None
        self.target_vertices = None
        self.perspective_transformer = None
        if pixel_vertices is None or target_vertices is None:
            return
        pixels = np.asarray(pixel_vertices, dtype=np.float32).reshape(4, 2)
        targets = np.asarray(target_vertices, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(pixels).all() or not np.isfinite(targets).all():
            raise ValueError("Los puntos de calibración deben ser finitos.")
        if abs(cv2.contourArea(pixels)) < 10:
            raise ValueError("Los puntos de imagen no forman un área válida.")
        if abs(cv2.contourArea(targets)) < 1:
            raise ValueError("Los puntos en metros no forman un área válida.")
        homography = cv2.getPerspectiveTransform(pixels, targets)
        if not np.isfinite(homography).all():
            raise ValueError("La homografía calculada no es válida.")
        if abs(np.linalg.det(homography)) < 1e-12:
            raise ValueError("La homografía es singular.")
        self.pixel_vertices = pixels
        self.target_vertices = targets
        self.perspective_transformer = homography
        self.enabled = True

    def transform_point(self, point):
        if not self.enabled or point is None:
            return None
        p = np.asarray(point, dtype=np.float32).reshape(2)
        if not np.isfinite(p).all():
            return None
        inside = cv2.pointPolygonTest(self.pixel_vertices, (float(p[0]), float(p[1])), False)
        if inside < 0:
            return None
        transformed = cv2.perspectiveTransform(p.reshape(1, 1, 2), self.perspective_transformer).reshape(2)
        if not np.isfinite(transformed).all():
            return None
        return transformed.astype(float)

    def add_transformed_position_to_tracks(self, tracks):
        for category, object_tracks in tracks.items():
            for frame_tracks in object_tracks:
                for item in frame_tracks.values():
                    item["position_transformed"] = self.transform_point(item.get("position"))

def load_view_transformer(calibration_path=CALIBRATION_PATH):
    try:
        with open(calibration_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        pixels = config.get("pixel_vertices")
        targets = config.get("field_vertices_m")
        if (not isinstance(pixels, list) or not isinstance(targets, list)
                or len(pixels) != 4 or len(targets) != 4):
            print(f"[Calibración] {Path(calibration_path).name} inválido. Se usará calibración por jugador.")
            return ViewTransformer()
        transformer = ViewTransformer(pixels, targets)
        print("[Calibración] Homografía estática cargada.")
        return transformer
    except FileNotFoundError:
        print(f"[Calibración] No existe {Path(calibration_path).name}. Se usará calibración por jugador.")
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"[Calibración] Configuración inválida: {exc}")
    return ViewTransformer()

def load_view_transformer_tramos(calibration_path=BASE_DIR / "calibracion_tramos.json"):
    if calibration_path.exists() and InterpolatedViewTransformer is not None:
        try:
            transformer = InterpolatedViewTransformer(calibration_path)
            if transformer.enabled:
                print("[Calibración] Modo tramos activo.")
                return transformer
        except Exception as exc:
            print(f"[Calibración] No se pudo cargar el modo tramos: {exc}")
    if InterpolatedViewTransformer is None and calibration_path.exists():
        print("[Calibración] Existe calibracion_tramos.json pero no se puede usar.")
    return load_view_transformer()

# ============================================================
# ARQUEROS Y ROLES
# ============================================================
def identify_goalkeepers(tracks):
    result = {TEAM_ARGENTINA: None, TEAM_ENGLAND: None}
    goalkeeper_counts = defaultdict(Counter)
    for frame_players in tracks["players"]:
        for track_id, player in frame_players.items():
            if player.get("role") != "goalkeeper":
                continue
            team = player.get("team", TEAM_NEUTRAL)
            if team in (TEAM_ARGENTINA, TEAM_ENGLAND):
                goalkeeper_counts[team][track_id] += 1
    for team in (TEAM_ARGENTINA, TEAM_ENGLAND):
        counts = goalkeeper_counts.get(team, Counter())
        if counts:
            result[team] = counts.most_common(1)[0][0]
    return result

# ============================================================
# POSESIÓN
# ============================================================
def estimate_possession(tracks):
    result = []
    last_team = TEAM_NEUTRAL
    total = min(len(tracks["players"]), len(tracks["ball"]))
    for frame_num in range(total):
        players = tracks["players"][frame_num]
        ball = tracks["ball"][frame_num].get(1, {})
        ball_pos = get_ball_position(ball)
        if ball_pos is not None:
            candidates = []
            for track_id, player in players.items():
                team = player.get("team", TEAM_NEUTRAL)
                if team not in (TEAM_ARGENTINA, TEAM_ENGLAND):
                    continue
                if player.get("role") == "referee":
                    continue
                pos = get_player_position(player)
                if pos is None:
                    continue
                distance = measure_distance(pos, ball_pos)
                _, player_height = player_bbox_size(player)
                candidates.append((distance, max(player_height, 1.0), team))
            if candidates:
                distance, height, team = min(candidates, key=lambda row: row[0])
                if distance <= 2.2 * height:
                    last_team = team
        result.append(last_team)
    while len(result) < len(tracks["players"]):
        result.append(last_team)
    return np.asarray(result, dtype=int)

# ============================================================
# ANÁLISIS TÁCTICO POR FRAME
# ============================================================
def analizar_jugada_en_frame(players, ball_position, possession, width, height):
    if ball_position is None:
        return None
    if possession not in (TEAM_ARGENTINA, TEAM_ENGLAND):
        return None
    defending_team = (TEAM_ENGLAND if possession == TEAM_ARGENTINA else TEAM_ARGENTINA)
    attackers = []
    defenders = []
    for track_id, player in players.items():
        team = player.get("team", TEAM_NEUTRAL)
        role = player.get("role", "field_player")
        if role == "referee":
            continue
        if team not in (TEAM_ARGENTINA, TEAM_ENGLAND):
            continue
        pos = get_player_position(player)
        if pos is None:
            continue
        item = {"id": track_id, "pos": pos, "role": role}
        if team == possession:
            attackers.append(item)
        elif team == defending_team:
            defenders.append(item)
    attackers = [p for p in attackers if p["role"] == "field_player"]
    defenders = [p for p in defenders if p["role"] == "field_player"]
    if not attackers or not defenders:
        return None
    heights = [player_bbox_size(players[p["id"]])[1] for p in attackers + defenders]
    scale = max(12.0, float(np.median(heights)))
    near_ball = [np.linalg.norm(d["pos"] - ball_position) for d in defenders]
    pressure = sum(distance < 2.0 * scale for distance in near_ball)
    free_receivers = []
    for attacker in attackers:
        distance_to_ball = np.linalg.norm(attacker["pos"] - ball_position)
        if distance_to_ball > 10.0 * scale:
            continue
        nearest_defender = min(np.linalg.norm(d["pos"] - attacker["pos"]) for d in defenders)
        if nearest_defender > 2.2 * scale:
            free_receivers.append((nearest_defender, attacker))
    if free_receivers:
        gap, receiver = max(free_receivers, key=lambda item: item[0])
        return ("RECEPTOR POTENCIALMENTE LIBRE", f"J{receiver['id']}: {gap:.0f}px del defensor más cercano", YELLOW)
    if pressure >= 2:
        return ("PRESIÓN DEFENSIVA POSIBLE", f"{pressure} defensores cerca del balón", GREEN)
    return None

# ============================================================
# FORMACIONES Y MÉTRICAS
# ============================================================
def formation_coordinate(player):
    position = tactical_position(player)
    if position is None:
        return None
    return float(position[1])

def split_into_formation_lines(values):
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) <= 1:
        return [values]
    gaps = np.diff(values)
    positive = gaps[gaps > 0]
    if len(positive) == 0:
        return [values]
    median_gap = float(np.median(positive))
    threshold = max(median_gap * LINE_GAP_FACTOR, median_gap + 8.0)
    split_indices = np.where(gaps > threshold)[0]
    if len(split_indices) > MAX_FORMATION_LINES - 1:
        largest = np.argsort(gaps[split_indices])[-(MAX_FORMATION_LINES - 1):]
        split_indices = np.sort(split_indices[largest])
    groups = np.split(values, split_indices + 1)
    return [group for group in groups if len(group) > 0]

def estimate_formation(players, team):
    values = []
    goalkeeper_present = False
    for player in players.values():
        if player.get("team") != team:
            continue
        role = player.get("role", "field_player")
        if role == "referee":
            continue
        if role == "goalkeeper":
            goalkeeper_present = True
            continue
        if role != "field_player":
            continue
        if not is_valid_player_detection(player):
            continue
        coordinate = formation_coordinate(player)
        if coordinate is not None:
            values.append(coordinate)
    if len(values) < 5:
        return f"Datos insuficientes ({len(values)})"
    groups = split_into_formation_lines(values)
    formation = "-".join(str(len(group)) for group in groups)
    if goalkeeper_present:
        formation += " + GK"
    return formation

def estimate_compactness(players, team):
    positions = []
    for player in players.values():
        if player.get("team") != team:
            continue
        if player.get("role") != "field_player":
            continue
        pos = tactical_position(player)
        if pos is not None:
            positions.append(pos)
    if len(positions) < 4:
        return "sin datos"
    points = np.asarray(positions, dtype=float)
    spread = float(np.mean(np.std(points, axis=0)))
    return f"{spread:.1f} px"

def calculate_team_metrics(players):
    result = {}
    for team in (TEAM_ARGENTINA, TEAM_ENGLAND):
        positions = []
        for player in players.values():
            if player.get("team") != team:
                continue
            if player.get("role") != "field_player":
                continue
            pos = tactical_position(player)
            if pos is not None:
                positions.append(pos)
        if not positions:
            result[team] = {"player_count": 0, "center_x_px": np.nan, "center_y_px": np.nan, "width_px": np.nan, "depth_px": np.nan}
            continue
        points = np.asarray(positions, dtype=float)
        result[team] = {"player_count": len(points), "center_x_px": float(np.mean(points[:, 0])), "center_y_px": float(np.mean(points[:, 1])), "width_px": float(np.ptp(points[:, 0])), "depth_px": float(np.ptp(points[:, 1]))}
    return result

def build_tactical_report(tracks, possession, fps=25.0):
    rows = []
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total = min(len(tracks["players"]), len(possession))
    for frame_num in range(total):
        players = tracks["players"][frame_num]
        metrics = calculate_team_metrics(players)
        row = {"frame": frame_num, "time_seconds": frame_num / fps, "estimated_possession_team": int(possession[frame_num])}
        for team in (TEAM_ARGENTINA, TEAM_ENGLAND):
            for metric_name, value in metrics[team].items():
                row[f"team_{team}_{metric_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)

def create_summary(report):
    rows = []
    for team in (TEAM_ARGENTINA, TEAM_ENGLAND):
        row = {"team": team}
        for column in ("player_count", "center_x_px", "center_y_px", "width_px", "depth_px"):
            name = f"team_{team}_{column}"
            if name in report.columns:
                row[f"average_{column}"] = report[name].mean()
        if not report.empty:
            row["frames_estimated_in_possession"] = int((report["estimated_possession_team"] == team).sum())
        else:
            row["frames_estimated_in_possession"] = 0
        rows.append(row)
    return pd.DataFrame(rows)

def create_tactical_map(tracks, output_path):
    plt.figure(figsize=(12, 7))
    for team in (TEAM_ARGENTINA, TEAM_ENGLAND):
        xs = []
        ys = []
        for frame_num in range(0, len(tracks["players"]), max(1, MAP_SAMPLE_EVERY)):
            for player in tracks["players"][frame_num].values():
                if player.get("team") != team:
                    continue
                if player.get("role") != "field_player":
                    continue
                pos = tactical_position(player)
                if pos is None:
                    continue
                xs.append(pos[0])
                ys.append(pos[1])
        if xs:
            plt.scatter(xs, ys, s=8, alpha=0.45, label=f"Equipo {team}")
    plt.title("Distribución espacial estabilizada")
    plt.xlabel("Coordenada horizontal (px)")
    plt.ylabel("Coordenada vertical (px)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ============================================================
# DIAGNÓSTICO DE TRACKING
# ============================================================
def save_tracking_debug(frame, players, output_path):
    debug = frame.copy()
    for track_id, player in players.items():
        bbox = player.get("bbox")
        if not bbox_is_valid(bbox):
            continue
        x1, y1, x2, y2 = map(int, bbox[:4])
        team = player.get("team", TEAM_NEUTRAL)
        role = player.get("role", "unknown")
        if role == "goalkeeper":
            color = GOALKEEPER_COLOR
        elif role == "referee":
            color = REFEREE_COLOR
        else:
            color = TEAM_COLORS.get(team, TEAM_COLORS[TEAM_NEUTRAL])
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id} E{team} {role}"
        cv2.putText(debug, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), debug)

# ============================================================
# DIBUJO DE JUGADORES
# ============================================================
def draw_player(frame, player, track_id, previous_positions):
    bbox = player.get("bbox")
    if not bbox_is_valid(bbox):
        return
    x1, y1, x2, y2 = map(int, bbox[:4])
    team = player.get("team", TEAM_NEUTRAL)
    role = player.get("role", "field_player")
    if role == "goalkeeper":
        color = GOALKEEPER_COLOR
        label = f"J{track_id} | E{team} | GK"
    elif role == "referee":
        color = REFEREE_COLOR
        label = f"J{track_id} | ARBITRO"
    elif team in (TEAM_ARGENTINA, TEAM_ENGLAND):
        color = TEAM_COLORS[team]
        label = f"J{track_id} | E{team}"
    else:
        color = TEAM_COLORS[TEAM_NEUTRAL]
        label = f"J{track_id} | NEUTRAL"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    position = get_player_position(player)
    if position is None:
        return
    foot = tuple(np.round(position).astype(int))
    cv2.circle(frame, foot, 4, color, -1)
    previous = previous_positions.get(track_id)
    if previous is not None:
        cv2.line(frame, previous, foot, color, 1, cv2.LINE_AA)
    previous_positions[track_id] = foot

# ============================================================
# DISTANCIAS ENTRE RIVALES
# ============================================================
def build_field_player_positions(players, frame_num=0):
    result = []
    for track_id, player in players.items():
        role = player.get("role", "field_player")
        team = player.get("team", TEAM_NEUTRAL)
        if role != "field_player" or team not in (TEAM_ARGENTINA, TEAM_ENGLAND):
            continue
        stable_pos = tactical_position(player)
        original_pos = get_player_position(player)
        if stable_pos is None or original_pos is None:
            continue
        escala = calcular_escala_local_por_jugador(player)
        result.append({"id": track_id, "team": team, "x": float(stable_pos[0]), "y": float(stable_pos[1]), "escala_m_px": escala})
    return result

def nearest_rival_pairs(field_players, max_draw_distance):
    pairs = []
    seen_pairs = set()
    for player in field_players:
        rivals = [r for r in field_players if r["team"] != player["team"]]
        if not rivals:
            continue
        def _pair_distance_px(other):
            return math.hypot(player["x"] - other["x"], player["y"] - other["y"])
        rival = min(rivals, key=_pair_distance_px)
        pair_key = tuple(sorted(((player["team"], player["id"]), (rival["team"], rival["id"]))))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        dpx = _pair_distance_px(rival)
        if dpx > max_draw_distance:
            continue
        e1 = player.get("escala_m_px")
        e2 = rival.get("escala_m_px")
        if e1 is not None and e2 is not None:
            escala = (e1 + e2) / 2.0
        elif e1 is not None:
            escala = e1
        elif e2 is not None:
            escala = e2
        else:
            escala = None
        if escala is not None:
            distancia_m = dpx * escala
            unit = "m"
            distance_val = distancia_m
        else:
            distance_val = dpx
            unit = "px"
        pairs.append((dpx, distance_val, unit, player, rival))
    pairs.sort(key=lambda row: row[1])
    return pairs[:5]

# ============================================================
# VIDEO TÁCTICO CON DISTANCIAS Y GOLEADOR
# ============================================================
def create_tactical_video(frames, tracks, team_ball_control, output_path, eventos_tacticos=None, view_transformer=None, autores_gol=None):
    if not frames:
        raise ValueError("No hay fotogramas para generar el video.")
    height, width = frames[0].shape[:2]
    fps = safe_fps(VIDEO_PATH)
    os.makedirs(os.path.dirname(str(output_path)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video: {output_path}")

    previous_positions = {}
    previous_ball_world = None
    previous_ball_frame = None
    previous_ball_px = None
    distancias_recorridas = defaultdict(float)  # track_id -> metros acumulados
    max_speed_ms = 55.0
    max_gap_seconds = 0.25
    total = min(len(frames), len(tracks["players"]), len(tracks["ball"]), len(team_ball_control))
    events = eventos_tacticos or []
    defensive_alerts_presentes = bool(events)

    # Índice rápido: frame de remate -> info del autor (metros de la jugada)
    autores_por_frame = {}
    for autor in (autores_gol or []):
        autores_por_frame[int(autor["fin_frame"])] = autor
        autores_por_frame[int(autor["inicio_frame"])] = autor

    # Historial de posesión por frame: quién tiene la pelota
    posesion_por_frame = {}
    for frame_num in range(total):
        players = tracks["players"][frame_num]
        ball_pos = get_ball_position(tracks["ball"][frame_num].get(1, {}))
        if ball_pos is None:
            posesion_por_frame[frame_num] = None
            continue
        min_dist = float('inf')
        poseedor = None
        for track_id, player in players.items():
            if player.get("role") == "referee":
                continue
            pos = get_player_position(player)
            if pos is None:
                continue
            dist = np.linalg.norm(pos - ball_pos)
            if dist < min_dist:
                min_dist = dist
                poseedor = track_id
        # Umbral de posesión: distancia menor a 1.5 veces la altura del jugador
        if poseedor is not None:
            _, altura_px = player_bbox_size(players[poseedor])
            if min_dist <= 1.5 * altura_px:
                posesion_por_frame[frame_num] = poseedor
            else:
                posesion_por_frame[frame_num] = None
        else:
            posesion_por_frame[frame_num] = None

    try:
        for frame_num in range(total):
            frame = frames[frame_num].copy()
            players = tracks["players"][frame_num]
            balls = tracks["ball"][frame_num]

            # Dibujar jugadores
            for track_id, player in players.items():
                draw_player(frame, player, track_id, previous_positions)

            # Acumular distancia recorrida por cada jugador
            for track_id, player in players.items():
                if player.get("role") == "referee":
                    continue
                
                distancia_metros = 0.0
                usa_homografia = False
                player_anterior = tracks["players"][frame_num - 1].get(track_id) if frame_num > 0 else None
                if player_anterior is not None:
                    # Primero usar las posiciones métricas ya calculadas por el
                    # calibrador (puede ser una homografía por tramos).
                    pt_act = player.get("position_transformed")
                    pt_ant = player_anterior.get("position_transformed")
                    if pt_act is not None and pt_ant is not None:
                        pt_act = np.asarray(pt_act, dtype=float).reshape(-1)[:2]
                        pt_ant = np.asarray(pt_ant, dtype=float).reshape(-1)[:2]
                        if np.isfinite(pt_act).all() and np.isfinite(pt_ant).all():
                            distancia_metros = float(np.linalg.norm(pt_act - pt_ant))
                            usa_homografia = True

                    # Compatibilidad con calibradores que no guardan posiciones
                    # transformadas dentro de cada track.
                    if not usa_homografia and view_transformer is not None and getattr(view_transformer, "enabled", False):
                        pos_actual = get_player_position(player)
                        pos_anterior = get_player_position(player_anterior)
                        if pos_actual is not None and pos_anterior is not None:
                            try:
                                pt_act = view_transformer.transform_point(pos_actual)
                                pt_ant = view_transformer.transform_point(pos_anterior)
                                if pt_act is not None and pt_ant is not None:
                                    distancia_metros = float(np.linalg.norm(np.asarray(pt_act)[:2] - np.asarray(pt_ant)[:2]))
                                    usa_homografia = np.isfinite(distancia_metros)
                            except (TypeError, ValueError, AttributeError):
                                usa_homografia = False

                    # Sin homografía fiable: compensar paneo y estimar escala con
                    # la altura de ambos frames, en lugar de medir píxeles crudos.
                    if not usa_homografia:
                        escala_actual = calcular_escala_local_por_jugador(player)
                        escala_anterior = calcular_escala_local_por_jugador(player_anterior)
                        escalas = [v for v in (escala_actual, escala_anterior) if v is not None and np.isfinite(v)]
                        pos_actual = player.get("position_adjusted")
                        pos_anterior = player_anterior.get("position_adjusted")
                        if not (pos_actual is not None and pos_anterior is not None):
                            pos_actual = get_player_position(player)
                            pos_anterior = get_player_position(player_anterior)
                        if escalas and pos_actual is not None and pos_anterior is not None:
                            escala = float(np.mean(escalas))
                            distancia_metros = float(np.linalg.norm(np.asarray(pos_actual, dtype=float)[:2] - np.asarray(pos_anterior, dtype=float)[:2]) * escala)

                    # Límite físico por tiempo real: evita saltos de identidad o
                    # detecciones erróneas sin descartar carreras rápidas a 25/30 FPS.
                    max_por_frame = max_speed_ms / max(1.0, fps)
                    if np.isfinite(distancia_metros) and 0 < distancia_metros <= max_por_frame:
                        distancias_recorridas[track_id] += distancia_metros

            ball = balls.get(1, {})
            ball_pos = get_ball_position(ball)
            if ball_pos is not None:
                center = tuple(np.round(ball_pos).astype(int))
                cv2.circle(frame, center, 7, YELLOW, 2)
                cv2.putText(frame, "BALON", (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)

            possession = int(team_ball_control[frame_num])
            analysis = analizar_jugada_en_frame(players, ball_pos, possession, width, height)
            formation_1 = estimate_formation(players, TEAM_ARGENTINA)
            formation_2 = estimate_formation(players, TEAM_ENGLAND)
            compact_1 = estimate_compactness(players, TEAM_ARGENTINA)
            compact_2 = estimate_compactness(players, TEAM_ENGLAND)
            field_players = build_field_player_positions(players, frame_num=frame_num)

            pairs = nearest_rival_pairs(field_players, max_draw_distance=width * 0.105)
            for _, distance, unit, player, rival in pairs:
                p1 = (int(player["x"]), int(player["y"]))
                p2 = (int(rival["x"]), int(rival["y"]))
                cv2.line(frame, p1, p2, YELLOW, 1, cv2.LINE_AA)
                label = f"E{player['team']}-J{player['id']} vs E{rival['team']}-J{rival['id']}: {distance:.1f} {unit}"
                cx = (p1[0] + p2[0]) // 2
                cy = (p1[1] + p2[1]) // 2
                cv2.putText(frame, label[:95], (max(5, min(width - 400, cx)), max(18, min(height - 5, cy))), cv2.FONT_HERSHEY_SIMPLEX, 0.36, YELLOW, 1, cv2.LINE_AA)

            # Determinar poseedor actual
            poseedor_actual = posesion_por_frame.get(frame_num)
            dist_total_poseedor = 0.0
            if poseedor_actual is not None:
                dist_total_poseedor = distancias_recorridas.get(poseedor_actual, 0.0)

            speed_ms = None
            speed_kmh = None
            metodo_velocidad = ""

            if (view_transformer is not None and getattr(view_transformer, "enabled", False)):
                ball_world = None
                try:
                    ball_world = view_transformer.transform_point(ball_pos)
                except Exception:
                    pass
                if ball_world is not None:
                    ball_world = np.asarray(ball_world, dtype=float).reshape(-1)[:2]
                    if (ball_world.size == 2 and np.isfinite(ball_world).all()):
                        if (previous_ball_world is not None and previous_ball_frame is not None):
                            frames_elapsed = frame_num - previous_ball_frame
                            dt = frames_elapsed / fps
                            if frames_elapsed > 0 and 0 < dt <= max_gap_seconds:
                                distance_m = float(np.linalg.norm(ball_world - previous_ball_world))
                                candidate_speed = distance_m / dt
                                if (np.isfinite(candidate_speed) and 0 <= candidate_speed <= max_speed_ms):
                                    speed_ms = candidate_speed
                                    speed_kmh = candidate_speed * 3.6
                                    metodo_velocidad = "homografía"
                                else:
                                    previous_ball_world = None
                                    previous_ball_frame = None
                        if ball_world is not None:
                            previous_ball_world = ball_world.copy()
                            previous_ball_frame = frame_num

            if speed_ms is None and ball_pos is not None:
                escala_balon = estimar_escala_balon_por_jugador_cercano(players, ball_pos)
                if escala_balon is not None:
                    if previous_ball_px is not None and previous_ball_frame is not None:
                        frames_elapsed = frame_num - previous_ball_frame
                        dt = frames_elapsed / fps
                        if frames_elapsed > 0 and 0 < dt <= max_gap_seconds:
                            dist_px = float(np.linalg.norm(np.asarray(ball_pos) - np.asarray(previous_ball_px)))
                            dist_m = dist_px * escala_balon
                            candidate_speed = dist_m / dt
                            if (np.isfinite(candidate_speed) and 0 <= candidate_speed <= max_speed_ms):
                                speed_ms = candidate_speed
                                speed_kmh = candidate_speed * 3.6
                                metodo_velocidad = "ref. jugador"
                    previous_ball_px = np.asarray(ball_pos, dtype=float).copy()
                    previous_ball_frame = frame_num
                else:
                    previous_ball_px = None
                    previous_ball_frame = None

            if speed_ms is not None:
                speed_text = f"Velocidad balón ({metodo_velocidad}): {speed_ms:.1f} m/s ({speed_kmh:.1f} km/h)"
            else:
                speed_text = "Velocidad balón: esperando datos..."

            # Jugador más cercano al balón (para mostrar su distancia)
            jugador_cercano_id = None
            if ball_pos is not None and field_players:
                min_dist = float('inf')
                for p in field_players:
                    dist = math.hypot(p["x"] - ball_pos[0], p["y"] - ball_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        jugador_cercano_id = p["id"]

            if ball_pos is not None:
                near_ball = []
                for player in field_players:
                    distance_px = math.hypot(player["x"] - ball_pos[0], player["y"] - ball_pos[1])
                    near_ball.append((distance_px, player))
                near_ball.sort(key=lambda item: item[0])
                for i, (distance_px, player) in enumerate(near_ball[:3]):
                    label = f"E{player['team']}-J{player['id']} balón: {distance_px:.0f} px"
                    escala = player.get("escala_m_px")
                    if escala is not None:
                        dist_m = distance_px * escala
                        label = f"E{player['team']}-J{player['id']} balón: {dist_m:.1f} m"
                    cv2.putText(frame, label, (max(10, width - 300), 35 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, YELLOW, 1, cv2.LINE_AA)

            if possession in (TEAM_ARGENTINA, TEAM_ENGLAND):
                possession_text = f"Posesion estimada: Equipo {possession}"
            else:
                possession_text = "Posesion: sin determinar"

            panel_lines = [
                ("ANALISIS TACTICO (ESTIMACIONES)", WHITE),
                (possession_text, TEAM_COLORS.get(possession, WHITE)),
                (f"E1 formacion: {formation_1}", TEAM_COLORS[TEAM_ARGENTINA]),
                (f"E2 formacion: {formation_2}", TEAM_COLORS[TEAM_ENGLAND]),
                (f"E1 dispersion: {compact_1}", TEAM_COLORS[TEAM_ARGENTINA]),
                (f"E2 dispersion: {compact_2}", TEAM_COLORS[TEAM_ENGLAND]),
                ("Distancias: homografia si valida; si no, escala estimada", WHITE),
                ("MODO REVISION: eventos candidatos" if defensive_alerts_presentes else "MODO REVISION: sin eventos en este tramo", WHITE),
                (speed_text, YELLOW),
            ]

            if poseedor_actual is not None:
                panel_lines.append((f"J{poseedor_actual} (poseedor) recorrió: {dist_total_poseedor:.1f} m", YELLOW))
            elif jugador_cercano_id is not None:
                dist_total = distancias_recorridas.get(jugador_cercano_id, 0.0)
                panel_lines.append((f"J{jugador_cercano_id} recorrió: {dist_total:.1f} m", YELLOW))

            panel_w = min(720, width - 20)
            panel_h = 210 + (22 if (poseedor_actual is not None or jugador_cercano_id is not None) else 0)
            cv2.rectangle(frame, (10, 10), (10 + panel_w, min(height - 5, 10 + panel_h)), DARK, cv2.FILLED)
            y = 34
            for line, color in panel_lines:
                cv2.putText(frame, line[:100], (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
                y += 22

            time_s = frame_num / fps
            active_event = next((event for event in events if float(event.get("inicio_seconds", 1e12)) <= time_s <= float(event.get("fin_seconds", -1)) + 0.65), None)

            if active_event:
                tipo = str(active_event.get("tipo_evento", "evento")).lower()
                etiquetas = {
                    "intercepcion_defensiva_posible": "POSIBLE INTERCEPCIÓN",
                    "pase_filtrado_posible": "POSIBLE PASE FILTRADO",
                    "error_defensivo_posible": "RECEPTOR CON ESPACIO",
                    "pase_completado_posible": "POSIBLE PASE COMPLETADO",
                    "receptor_libre_posible": "POSIBLE RECEPTOR LIBRE",
                    "tiro_al_arco": "REMATE POSIBLE",
                    "conduccion": "CONDUCCIÓN DETECTADA",
                    "recepcion": "RECEPCIÓN POSIBLE",
                    "perdida": "POSIBLE PÉRDIDA DE BALÓN",
                    "gol_probable": "POSIBLE GOL: REVISAR VIDEO",
                    "fin_ataque": "FIN DE ATAQUE",
                }
                title = etiquetas.get(tipo, "SECUENCIA DE ATAQUE")

                def _fmt(valor, unidad):
                    if valor in ("", None):
                        return None
                    try:
                        numero = float(valor)
                        if not np.isfinite(numero):
                            return None
                        return f"{numero:.1f} {unidad}"
                    except (TypeError, ValueError):
                        return None

                dist_pase_m = _fmt(active_event.get("distancia_pase_m"), "m")
                dist_pase_px = _fmt(active_event.get("distancia_pase_px"), "px")
                dist_def_m = _fmt(active_event.get("distancia_defensor_receptor_m"), "m")
                dist_def_px = _fmt(active_event.get("distancia_defensor_receptor_px"), "px")
                vel_m = _fmt(active_event.get("velocidad_pase_m_s"), "m/s")
                vel_kmh = _fmt(active_event.get("velocidad_pase_km_h"), "km/h")
                vel_tiro_m = _fmt(active_event.get("velocidad_tiro_m_s"), "m/s")
                vel_tiro_kmh = _fmt(active_event.get("velocidad_tiro_km_h"), "km/h")

                lineas = []
                detalle_evento = str(active_event.get("detalle", "")).strip()
                if detalle_evento:
                    lineas.append(detalle_evento)
                metros_conduccion = _fmt(active_event.get("metros_conduccion"), "m conducidos")
                if metros_conduccion:
                    lineas.append(f"Conducción del jugador: {metros_conduccion}")
                metros_equipo = _fmt(active_event.get("metros_conduccion_equipo"), "m acumulados")
                if metros_equipo:
                    lineas.append(f"Conducción del equipo en la jugada: {metros_equipo}")
                if vel_tiro_m:
                    lineas.append(f"Velocidad del tiro: {vel_tiro_m}" + (f" ({vel_tiro_kmh})" if vel_tiro_kmh else ""))
                if dist_pase_m:
                    lineas.append(f"Distancia del pase: {dist_pase_m}")
                elif dist_pase_px:
                    lineas.append(f"Distancia del pase: {dist_pase_px} (sin calibrar)")
                if dist_def_m:
                    lineas.append(f"Defensor-receptor: {dist_def_m}")
                elif dist_def_px:
                    lineas.append(f"Defensor-receptor: {dist_def_px} (sin calibrar)")
                if vel_m:
                    lineas.append(f"Velocidad del pase: {vel_m}" + (f" ({vel_kmh})" if vel_kmh else ""))

                # Si es tiro al arco, mostrar la distancia del tirador y del autor del gol
                if tipo == "tiro_al_arco" and active_event.get("tirador_id") is not None:
                    tirador_id = active_event["tirador_id"]
                    dist_tirador = distancias_recorridas.get(tirador_id, 0.0)
                    lineas.append(f"Distancia del tirador J{tirador_id}: {dist_tirador:.1f} m")
                    info_autor = autores_por_frame.get(int(active_event.get("inicio_frame", -1)))
                    if info_autor is not None and info_autor.get("metros_autor") is not None:
                        lineas.append(
                            f"Recorrido del autor J{info_autor['tirador_id']}: "
                            f"{info_autor['metros_autor']:.1f} m en la jugada"
                        )
                        pases = info_autor.get("pases_previos", 0)
                        jugadores = info_autor.get("jugadores_previos", [])
                        if pases or jugadores:
                            lineas.append(
                                f"Secuencia: {pases} pase(s) previo(s); jugadores {jugadores}"
                            )

                ey = min(max(200, height - 90), height - 70)
                alto_caja = 30 + 22 * (1 + len(lineas))
                cv2.rectangle(frame, (10, ey), (min(width - 10, 950), min(height - 5, ey + alto_caja)), DARK, cv2.FILLED)
                cv2.putText(frame, title[:80], (20, ey + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, YELLOW, 2, cv2.LINE_AA)
                y_txt = ey + 44
                for linea in lineas[:5]:
                    cv2.putText(frame, linea[:100], (20, min(height - 8, y_txt)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
                    y_txt += 22
            elif analysis is not None:
                title, detail, color = analysis
                ey = min(220, max(10, height - 70))
                cv2.rectangle(frame, (10, ey), (min(width - 10, 740), min(height - 5, ey + 52)), DARK, cv2.FILLED)
                cv2.putText(frame, title[:70], (20, ey + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
                cv2.putText(frame, detail[:100], (20, min(height - 8, ey + 43)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

            minutes = int(time_s // 60)
            seconds = int(time_s % 60)
            cv2.putText(frame, f"{minutes:02d}:{seconds:02d} | Frame {frame_num + 1}/{total}", (max(10, width - 310), height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 2, cv2.LINE_AA)
            writer.write(frame)

            if (frame_num + 1) % 100 == 0:
                print(f"Video tactico: {frame_num + 1}/{total}")

    finally:
        writer.release()

    print(f"Video tactico generado: {output_path}")

def load_auto_calibrador():
    auto = AutoCalibrador()
    if auto.enabled:
        return auto
    print("[Calibración] AutoCalibrador no disponible; uso calibración por jugador.")
    return ViewTransformer()

# ============================================================
# PROCESO PRINCIPAL
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STUBS_DIR.mkdir(parents=True, exist_ok=True)
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"No se encontró el video: {VIDEO_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    print("=" * 60)
    print("ANÁLISIS TÁCTICO DE FÚTBOL - MODO HÍBRIDO CON GOLEADOR")
    print("=" * 60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Clip: {CLIP_NAME}")

    fps = safe_fps(VIDEO_PATH)
    print("\n1/6 - Cargando video...")
    frames = read_video(VIDEO_PATH)
    print(f"Fotogramas: {len(frames)}")
    print(f"FPS: {fps:.2f}")

    print("\n2/6 - Detectando y siguiendo objetos...")
    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(frames, read_from_stub=False, stub_path=str(TRACKS_STUB))
    if len(tracks["players"]) != len(frames):
        raise RuntimeError(f"La cantidad de fotogramas no coincide con el tracking: {len(frames)} vs {len(tracks['players'])}")
    tracker.add_position_to_tracks(tracks)
    stabilize_track_roles(tracks)

    if len(frames) > 105:
        save_tracking_debug(frames[105], tracks["players"][105], DEBUG_TRACKING_PATH)
        print(f"Diagnóstico guardado: {DEBUG_TRACKING_PATH}")

    print("\n3/6 - Separando los equipos...")
    team_assigner = TeamAssigner()
    trained = team_assigner.assign_team_color_multi_frame(frames, tracks, max_frames=60)
    if trained:
        votes = collect_team_votes(tracks, team_assigner, frames)
        resolved = resolve_track_teams(votes)
        apply_resolved_teams(tracks, resolved)
    else:
        apply_resolved_teams(tracks, {})
    stabilize_track_roles(tracks)
    goalkeeper_map = identify_goalkeepers(tracks)
    print("Arqueros clasificados por el detector:", goalkeeper_map)

    print("\n4/6 - Estabilizando coordenadas y cargando calibración...")
    camera_estimator = CameraMovementEstimator(frames[0])
    camera_movements = camera_estimator.get_camera_movement(frames, read_from_stub=False, stub_path=str(CAMERA_STUB))
    print(f"Desplazamientos de cámara: {len(camera_movements)}")
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movements)

    view_transformer = load_auto_calibrador()
    if hasattr(view_transformer, "calibrar_frames"):
        calibraciones_validas = view_transformer.calibrar_frames(frames)
        if not calibraciones_validas:
            print("[Calibración] El calibrador automático no consiguió homografías válidas; se usará calibración por jugador.")
            view_transformer = ViewTransformer()
    print(f"[Calibración] view_transformer usado -> tipo={type(view_transformer).__name__}, enabled={getattr(view_transformer, 'enabled', None)}")
    view_transformer.add_transformed_position_to_tracks(tracks)

    referee_tracks = set()
    for frame_players in tracks["players"]:
        for track_id, player in frame_players.items():
            if player.get("role") == "referee":
                referee_tracks.add(track_id)
    print("Tracks de árbitros:", referee_tracks)

    print("\n5/6 - Calculando métricas y eventos...")
    possession = estimate_possession(tracks)
    # Reconstrucción automática de ataques: conducciones, pases, recepciones,
    # remates y cambios de posesión. No requiere minutos de gol manuales.
    defensive_alerts = detectar_alertas_defensivas(
        tracks=tracks,
        fps=fps,
        salida_csv=None,
        tiempos_goles=None,
        equipo_en_posesion=possession,
        view_transformer=view_transformer,
        bbox_arco=None,
    ) or []
    print(f"[Ataques] Eventos automáticos detectados: {len(defensive_alerts)}")
    resumen_ataques = resumir_secuencias_ataque(defensive_alerts)
    for n, ataque in enumerate(resumen_ataques[:20], start=1):
        print(f"[Ataque {n}] {ataque}")

    # Métricas principales por cada posible remate. El ID de tracking no es un nombre real;
    # para nombres reales hace falta reconocer dorsales o cargar una plantilla del partido.
    eventos_remate = [e for e in defensive_alerts if e.get("tipo_evento") == "tiro_al_arco"]
    print("\n=== MÉTRICAS DEL POSIBLE REMATADOR ===")
    if not eventos_remate:
        print("No se detectaron posibles remates en este video.")
    else:
        vistos = set()
        for evento in eventos_remate:
            jugador_id = evento.get("tirador_id")
            if jugador_id in (None, ""):
                print("Remate posible detectado, pero no se pudo asociar a un jugador.")
                continue
            frame_remate = int(evento.get("inicio_frame", 0))
            clave = (jugador_id, frame_remate // max(1, int(2 * fps)))
            if clave in vistos:
                continue
            vistos.add(clave)
            # Ventana provisional: 15 segundos previos al posible remate.
            # No se afirma que sea toda la jugada ni que el remate termine en gol.
            inicio_tramo = max(0, frame_remate - int(15 * fps))
            fin_tramo = min(len(tracks.get("players", [])) - 1, int(evento.get("fin_frame", frame_remate)))
            metricas = metricas_jugador_con_sin_pelota(
                tracks, jugador_id, inicio_tramo, fin_tramo, fps=fps,
                view_transformer=view_transformer
            )
            inicio_seg = float(evento.get("inicio_seconds", frame_remate / fps))
            minuto, segundo = int(inicio_seg // 60), int(inicio_seg % 60)
            # Cuenta pases inferidos dentro de la ventana de análisis.
            pases = [e for e in defensive_alerts
                     if e.get("tipo_evento") == "pase_completado_posible"
                     and inicio_tramo / fps <= float(e.get("inicio_seconds", 0)) <= fin_tramo / fps]
            velocidad_kmh = evento.get("velocidad_tiro_km_h")
            print("-" * 58)
            print(f"[{minuto:02d}:{segundo:02d}] Jugador ID {jugador_id} (nombre real no identificado)")
            print(f"  Pases estimados en la secuencia/ventana: {len(pases)}")
            print(f"  Distancia total del jugador: {metricas['total_m']} m")
            print(f"  Distancia con pelota: {metricas['con_pelota_m']} m")
            print(f"  Distancia sin pelota: {metricas['sin_pelota_m']} m")
            print(f"  Método de distancia: {metricas['metodo']}")
            if velocidad_kmh is not None:
                print(f"  Velocidad estimada del balón: {velocidad_kmh:.1f} km/h (solo válida si la calibración es correcta)")
            else:
                print("  Velocidad del tiro en km/h: no disponible; falta una trayectoria métrica calibrada del balón")
            print("  Estado: posible remate; el sistema no confirma automáticamente que haya sido gol.")
        print("\nNota: el intervalo es de ~15 s previos al remate. Los pases y la posesión son estimaciones automáticas.")
        print("El código muestra ID de seguimiento, no nombre real: para eso hace falta leer el dorsal o asociar la plantilla.")

    # === NUEVO: metros del autor del gol (general para cualquier variante) ===
    autores_gol = distancia_autor_gol(
        tracks=tracks,
        events=defensive_alerts,
        fps=fps,
        view_transformer=view_transformer,
    )
    print("\n=== DISTANCIA RECORRIDA POR EL AUTOR DEL GOL ===")
    if not autores_gol:
        print("No se pudo asociar un autor de remate a una secuencia.")
    else:
        for autor in autores_gol:
            m, s = int(autor["inicio_seconds"] // 60), int(autor["inicio_seconds"] % 60)
            print("-" * 58)
            print(f"[{m:02d}:{s:02d}] Autor J{autor['tirador_id']} (equipo {autor['equipo']})")
            print(f"  Ventana de jugada: frames {autor['inicio_frame']}–{autor['fin_frame']}")
            print(f"  Distancia recorrida por el autor: {autor['metros_autor']} m")
            print(f"  Método: {autor['metodo']}")
            print(f"  Pases previos en la secuencia: {autor['pases_previos']}")
            print(f"  Jugadores involucrados antes del remate: {autor['jugadores_previos']}")

    report = build_tactical_report(tracks, possession, fps=fps)
    report.to_csv(REPORT_PATH, index=False)
    create_summary(report).to_csv(SUMMARY_PATH, index=False)
    create_tactical_map(tracks, MAP_PATH)

    create_tactical_video(
        frames=frames, tracks=tracks, team_ball_control=possession,
        output_path=TACTICAL_VIDEO_PATH, eventos_tacticos=defensive_alerts,
        view_transformer=view_transformer,
        autores_gol=autores_gol,
    )

    print("\n6/6 - Resultados")
    print(f"Video táctico: {TACTICAL_VIDEO_PATH}")
    print(f"Informe:       {REPORT_PATH}")
    print(f"Resumen:       {SUMMARY_PATH}")
    print(f"Mapa:          {MAP_PATH}")
    print(f"Diagnóstico:   {DEBUG_TRACKING_PATH}")
    print("\nPrimeras filas del informe:")
    print(report.head(10).to_string(index=False))

if __name__ == "__main__":
    main()