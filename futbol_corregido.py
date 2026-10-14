# ============================================================
# FUTBOL_CORREGIDO.PY
# Pipeline de análisis táctico de fútbol.
# ============================================================

import os
import sys
import json
import math
import pickle
from pathlib import Path
from collections import Counter, defaultdict, deque
from auto_calibrador import AutoCalibrador  # <-- AGREGAR

from view_transformer_tramos import InterpolatedViewTransformer
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

from defensee import detectar_alertas_defensivas


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_VIDEO_PATH = Path(
    r"C:\Users\Matias\Downloads\chivas_penal.mp4"
)

VIDEO_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH

CLIP_NAME = VIDEO_PATH.stem

MODEL_PATH = BASE_DIR / "models" / "best.pt"
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
    return float(np.linalg.norm(np.asarray(p1, dtype=float) -
                                np.asarray(p2, dtype=float)))


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

    return (
        float(bbox[2] - bbox[0]),
        float(bbox[3] - bbox[1]),
    )


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
    def __init__(self, model_path):
        self.model = YOLO(str(model_path))
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        detections = []

        for start in range(0, len(frames), DETECTION_BATCH_SIZE):
            batch = frames[start:start + DETECTION_BATCH_SIZE]

            results = self.model.predict(
                batch,
                conf=DETECTION_CONFIDENCE,
                verbose=False,
            )

            detections.extend(results)

        return detections

    def get_object_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None,
    ):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            try:
                with open(stub_path, "rb") as file:
                    cached = pickle.load(file)

                if (
                    len(cached.get("players", [])) == len(frames)
                    and len(cached.get("ball", [])) == len(frames)
                ):
                    print("[Tracker] Stub válido.")
                    return cached

            except Exception as exc:
                print(f"[Tracker] No se pudo leer el stub: {exc}")

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            names = {
                int(class_id): str(name).strip().lower()
                for class_id, name in detection.names.items()
            }

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            dets = sv.Detections.from_ultralytics(detection)

            tracked_dets = self.tracker.update_with_detections(dets)

            # Jugadores, arqueros y árbitros con ID de seguimiento.
            for item in tracked_dets:
                bbox = item[0].tolist()
                class_id_value = item[3]
                track_id_value = item[4]

                if class_id_value is None or track_id_value is None:
                    continue

                class_id = int(class_id_value)
                track_id = int(track_id_value)
                class_name = names.get(class_id, "")

                if not bbox_is_valid(bbox):
                    continue

                if class_name == "player":
                    role = "field_player"

                elif class_name == "goalkeeper":
                    role = "goalkeeper"

                elif class_name == "referee":
                    role = "referee"

                else:
                    continue

                info = {
                    "bbox": bbox,
                    "role": role,
                    "class_name": class_name,
                }

                tracks["players"][frame_num][track_id] = info

                if role == "referee":
                    tracks["referees"][frame_num][track_id] = info.copy()

            # Balón: se conserva la detección con mayor confianza.
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

                confidence = (
                    float(confidence)
                    if confidence is not None
                    else 0.0
                )

                ball_candidates.append((confidence, bbox))

            if ball_candidates:
                confidence, bbox = max(
                    ball_candidates,
                    key=lambda value: value[0],
                )

                tracks["ball"][frame_num][1] = {
                    "bbox": bbox,
                    "class_name": "ball",
                    "confidence": confidence,
                }

        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)

            with open(stub_path, "wb") as file:
                pickle.dump(tracks, file)

        return tracks

    def add_position_to_tracks(self, tracks):
        for category, category_tracks in tracks.items():
            for frame_tracks in category_tracks:
                for info in frame_tracks.values():
                    bbox = info.get("bbox")

                    if not bbox_is_valid(bbox):
                        continue

                    if category == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)

                    info["position"] = tuple(position)

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        if not bbox_is_valid(bbox):
            return frame

        x_center, _ = get_center_of_bbox(bbox)
        y2 = int(bbox[3])
        width = max(8, int(get_bbox_width(bbox)))

        cv2.ellipse(
            frame,
            (x_center, y2),
            (max(8, width // 2), max(4, int(width * 0.18))),
            0,
            0,
            360,
            color,
            2,
        )

        if track_id is not None:
            text = str(track_id)
            cv2.putText(
                frame,
                text,
                (x_center - 10, y2 + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        if not bbox_is_valid(bbox):
            return frame

        x, y = get_center_of_bbox(bbox)
        y = int(bbox[1])

        points = np.array(
            [[x, y], [x - 8, y - 14], [x + 8, y - 14]],
            dtype=np.int32,
        )

        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 1)

        return frame


# ============================================================
# ROLES ESTABLES
# ============================================================

def stabilize_track_roles(tracks):
    """
    Consolida el rol de cada ID usando sus clasificaciones observadas.
    No inventa arqueros por estar cerca de un borde de la imagen.
    """
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

        # Si hay conflicto fuerte entre categorías, evitar inventar un rol.
        if total >= 3 and count / total < 0.50:
            role = "field_player"

        stable_roles[track_id] = role

    for frame_players in tracks["players"]:
        for track_id, player in frame_players.items():
            role = stable_roles.get(
                track_id,
                player.get("role", "field_player"),
            )

            player["role"] = role

            if role == "referee":
                player["team"] = TEAM_NEUTRAL
                player["team_color"] = REFEREE_COLOR

    return stable_roles


# ============================================================
# ASIGNACIÓN DE EQUIPOS
# ============================================================

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

        # Zona central del torso, evitando en lo posible fondo y pantalón.
        torso = crop[
            int(height * 0.18):int(height * 0.52),
            int(width * 0.20):int(width * 0.80),
        ]

        if torso.size == 0:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # Descarta sombras demasiado oscuras.
        pixels = pixels[pixels[:, 2] > 35]

        if len(pixels) < 10:
            return None

        hue = float(np.median(pixels[:, 0])) / 180.0
        saturation = float(np.median(pixels[:, 1])) / 255.0
        value = float(np.median(pixels[:, 2])) / 255.0

        # Atenuar el tono cuando casi no hay saturación.
        return np.array(
            [hue * saturation, saturation, value],
            dtype=np.float32,
        )

    def assign_team_color_multi_frame(
        self,
        frames,
        tracks,
        max_frames=60,
    ):
        total = min(len(frames), len(tracks["players"]))

        if total == 0:
            print("[Equipos] No hay fotogramas.")
            return False

        sample_indices = np.linspace(
            0,
            total - 1,
            num=min(max_frames, total),
            dtype=int,
        )

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
            print(
                "[Equipos] No hay suficientes muestras de color. "
                "Se dejarán neutrales las asignaciones dudosas."
            )
            return False

        colors = np.asarray(colors, dtype=np.float32)

        if len(np.unique(np.round(colors, 3), axis=0)) < 2:
            print("[Equipos] Los colores detectados son demasiado similares.")
            return False

        model = KMeans(
            n_clusters=2,
            n_init=15,
            random_state=42,
        )

        labels = model.fit_predict(colors)

        if len(np.unique(labels)) != 2:
            print("[Equipos] KMeans no pudo separar dos grupos.")
            return False

        self.kmeans = model
        self.team_colors = {
            1: model.cluster_centers_[0].copy(),
            2: model.cluster_centers_[1].copy(),
        }

        self.player_team_dict.clear()
        self.player_team_history.clear()

        print(
            f"[Equipos] KMeans entrenado con {len(colors)} "
            f"muestras de hasta {len(sample_indices)} fotogramas."
        )

        return True

    def get_player_team(
        self,
        frame,
        player_bbox,
        player_id,
        role="field_player",
    ):
        if role in ("goalkeeper", "referee", "neutral"):
            return TEAM_NEUTRAL

        if self.kmeans is None:
            return TEAM_NEUTRAL

        color = self.get_player_color(frame, player_bbox)

        if color is None:
            return self.player_team_dict.get(player_id, TEAM_NEUTRAL)

        prediction = int(
            self.kmeans.predict(color.reshape(1, -1))[0]
        ) + 1

        if player_id not in self.player_team_history:
            self.player_team_history[player_id] = deque(
                maxlen=self.history_size
            )

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

            team = team_assigner.get_player_team(
                frames[frame_num],
                player["bbox"],
                track_id,
                role,
            )

            votes[track_id].append(team)

    return votes


def resolve_track_teams(team_votes):
    resolved = {}

    for track_id, votes in team_votes.items():
        valid = [
            team for team in votes
            if team in (TEAM_ARGENTINA, TEAM_ENGLAND)
        ]

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
            player["team_color"] = TEAM_COLORS.get(
                team,
                TEAM_COLORS[TEAM_NEUTRAL],
            )


# ============================================================
# POSICIONES
# ============================================================

def get_player_position(player):
    """Pie del jugador en coordenadas originales de imagen."""
    bbox = player.get("bbox")

    if not bbox_is_valid(bbox):
        return None

    x1, y1, x2, y2 = map(float, bbox[:4])

    return np.array(
        [(x1 + x2) / 2.0, y2],
        dtype=float,
    )


def get_ball_position(ball):
    bbox = ball.get("bbox")

    if bbox_is_valid(bbox):
        x1, y1, x2, y2 = map(float, bbox[:4])

        return np.array(
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            dtype=float,
        )

    position = ball.get("position")

    if position is not None and len(position) >= 2:
        point = np.asarray(position[:2], dtype=float)

        if np.all(np.isfinite(point)):
            return point

    return None


def tactical_position(player):
    """Posición estabilizada en píxeles; no está expresada en metros."""
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
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }

    def get_camera_movement(
        self,
        frames,
        read_from_stub=False,
        stub_path=None,
    ):
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
        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=400,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )

        for frame_num in range(1, len(frames)):
            gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            dx, dy = 0.0, 0.0

            if previous_points is not None and len(previous_points) >= 5:
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous_gray,
                    gray,
                    previous_points,
                    None,
                    **self.lk_params,
                )

                if new_points is not None and status is not None:
                    mask = status.reshape(-1) == 1

                    old_good = previous_points.reshape(-1, 2)[mask]
                    new_good = new_points.reshape(-1, 2)[mask]

                    if len(old_good) >= 5:
                        displacement = new_good - old_good

                        # La mediana es más resistente a jugadores en movimiento
                        # que la media de todos los puntos ópticos.
                        dx, dy = np.median(displacement, axis=0).tolist()

                    previous_points = new_points[status.reshape(-1) == 1]

            movements.append((float(dx), float(dy)))

            if previous_points is None or len(previous_points) < 20:
                previous_points = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=400,
                    qualityLevel=0.01,
                    minDistance=8,
                    blockSize=7,
                )

            previous_gray = gray

        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)

            with open(stub_path, "wb") as file:
                pickle.dump(movements, file)

        return movements

    def add_adjust_positions_to_tracks(self, tracks, movements):
        """
        Convierte el movimiento entre frames en desplazamiento acumulado.
        La posición estabilizada = posición original - movimiento acumulado.
        """
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

                    item["position_adjusted"] = (
                        float(position[0] - dx),
                        float(position[1] - dy),
                    )


# ============================================================
# CALIBRACIÓN DE CANCHA
# ============================================================

class ViewTransformer:
    """
    Homografía de píxeles de la imagen original a metros.

    pixel_vertices:
        Cuatro esquinas del área calibrada en el video original.

    target_vertices:
        Las mismas cuatro esquinas, en el mismo orden, en metros.
    """

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

        inside = cv2.pointPolygonTest(
            self.pixel_vertices,
            (float(p[0]), float(p[1])),
            False,
        )

        if inside < 0:
            return None

        transformed = cv2.perspectiveTransform(
            p.reshape(1, 1, 2),
            self.perspective_transformer,
        ).reshape(2)

        if not np.isfinite(transformed).all():
            return None

        return transformed.astype(float)

    def add_transformed_position_to_tracks(self, tracks):
        """
        IMPORTANTE: la homografía se aplica a la posición ORIGINAL.
        Nunca se usa position_adjusted porque pertenece a un sistema
        de coordenadas estabilizado diferente.
        """
        for category, object_tracks in tracks.items():
            for frame_tracks in object_tracks:
                for item in frame_tracks.values():
                    item["position_transformed"] = self.transform_point(
                        item.get("position")
                    )


def load_view_transformer(calibration_path=CALIBRATION_PATH):
    """
    Carga calibracion_cancha.json si existe. Si no, devuelve un
    ViewTransformer deshabilitado (modo píxeles).
    """
    try:
        with open(calibration_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        pixels = config.get("pixel_vertices")
        targets = config.get("field_vertices_m")

        if (
            not isinstance(pixels, list)
            or not isinstance(targets, list)
            or len(pixels) != 4
            or len(targets) != 4
        ):
            print(
                f"[Calibración] No configurada. "
                f"Editá {Path(calibration_path).name}; se usarán píxeles."
            )
            return ViewTransformer()

        transformer = ViewTransformer(pixels, targets)

        print("[Calibración] Homografía estática cargada.")
        print("[Calibración] Distancias en metros habilitadas dentro del área.")

        return transformer

    except FileNotFoundError:
        print(
            f"[Calibración] No existe {Path(calibration_path).name}. "
            "Las distancias se mostrarán en píxeles."
        )

    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"[Calibración] Configuración inválida: {exc}")

    return ViewTransformer()

def load_view_transformer_tramos(
    calibration_path=BASE_DIR / "calibracion_tramos.json",
):
    """
    Si existe calibracion_tramos.json, usa homografia interpolada por frame.
    Si no, cae al calibracion_cancha.json de siempre.
    """
    if calibration_path.exists():
        transformer = InterpolatedViewTransformer(calibration_path)
        if transformer.enabled:
            print("[Calibración] Modo tramos activo.")
            return transformer

    print("[Calibración] Sin calibracion_tramos.json; uso el modo simple.")
    return load_view_transformer()


# ============================================================
# ARQUEROS Y ROLES
# ============================================================

def identify_goalkeepers(tracks):
    """
    No asigna arqueros según el extremo izquierdo o derecho de la imagen.
    Solo conserva los arqueros clasificados por el detector.
    """
    result = {
        TEAM_ARGENTINA: None,
        TEAM_ENGLAND: None,
    }

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
    """
    Estima posesión por proximidad al balón.
    No confirma control real cuando el balón está lejos de los jugadores.
    """
    result = []
    last_team = TEAM_NEUTRAL

    total = min(
        len(tracks["players"]),
        len(tracks["ball"]),
    )

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

                candidates.append(
                    (distance, max(player_height, 1.0), team)
                )

            if candidates:
                distance, height, team = min(
                    candidates,
                    key=lambda row: row[0],
                )

                # Evita asignar posesión si el balón está demasiado lejos.
                if distance <= 2.2 * height:
                    last_team = team

        result.append(last_team)

    while len(result) < len(tracks["players"]):
        result.append(last_team)

    return np.asarray(result, dtype=int)


# ============================================================
# ANÁLISIS TÁCTICO POR FRAME
# ============================================================

def analizar_jugada_en_frame(
    players,
    ball_position,
    possession,
    width,
    height,
):
    if ball_position is None:
        return None

    if possession not in (TEAM_ARGENTINA, TEAM_ENGLAND):
        return None

    defending_team = (
        TEAM_ENGLAND
        if possession == TEAM_ARGENTINA
        else TEAM_ARGENTINA
    )

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

        item = {
            "id": track_id,
            "pos": pos,
            "role": role,
        }

        if team == possession:
            attackers.append(item)
        elif team == defending_team:
            defenders.append(item)

    attackers = [
        p for p in attackers
        if p["role"] == "field_player"
    ]

    defenders = [
        p for p in defenders
        if p["role"] == "field_player"
    ]

    if not attackers or not defenders:
        return None

    heights = [
        player_bbox_size(players[p["id"]])[1]
        for p in attackers + defenders
    ]

    scale = max(12.0, float(np.median(heights)))

    near_ball = [
        np.linalg.norm(d["pos"] - ball_position)
        for d in defenders
    ]

    pressure = sum(distance < 2.0 * scale for distance in near_ball)

    free_receivers = []

    for attacker in attackers:
        distance_to_ball = np.linalg.norm(
            attacker["pos"] - ball_position
        )

        if distance_to_ball > 10.0 * scale:
            continue

        nearest_defender = min(
            np.linalg.norm(d["pos"] - attacker["pos"])
            for d in defenders
        )

        if nearest_defender > 2.2 * scale:
            free_receivers.append((nearest_defender, attacker))

    if free_receivers:
        gap, receiver = max(
            free_receivers,
            key=lambda item: item[0],
        )

        return (
            "RECEPTOR POTENCIALMENTE LIBRE",
            f"J{receiver['id']}: {gap:.0f}px del defensor más cercano",
            YELLOW,
        )

    if pressure >= 2:
        return (
            "PRESIÓN DEFENSIVA POSIBLE",
            f"{pressure} defensores cerca del balón; no confirma recuperación",
            GREEN,
        )

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
    threshold = max(
        median_gap * LINE_GAP_FACTOR,
        median_gap + 8.0,
    )

    split_indices = np.where(gaps > threshold)[0]

    if len(split_indices) > MAX_FORMATION_LINES - 1:
        largest = np.argsort(gaps[split_indices])[
            -(MAX_FORMATION_LINES - 1):
        ]

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
            result[team] = {
                "player_count": 0,
                "center_x_px": np.nan,
                "center_y_px": np.nan,
                "width_px": np.nan,
                "depth_px": np.nan,
            }
            continue

        points = np.asarray(positions, dtype=float)

        result[team] = {
            "player_count": len(points),
            "center_x_px": float(np.mean(points[:, 0])),
            "center_y_px": float(np.mean(points[:, 1])),
            "width_px": float(np.ptp(points[:, 0])),
            "depth_px": float(np.ptp(points[:, 1])),
        }

    return result


def build_tactical_report(tracks, possession, fps=25.0):
    rows = []

    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    total = min(len(tracks["players"]), len(possession))

    for frame_num in range(total):
        players = tracks["players"][frame_num]
        metrics = calculate_team_metrics(players)

        row = {
            "frame": frame_num,
            "time_seconds": frame_num / fps,
            "estimated_possession_team": int(possession[frame_num]),
        }

        for team in (TEAM_ARGENTINA, TEAM_ENGLAND):
            for metric_name, value in metrics[team].items():
                row[f"team_{team}_{metric_name}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def create_summary(report):
    rows = []

    for team in (TEAM_ARGENTINA, TEAM_ENGLAND):
        row = {"team": team}

        for column in (
            "player_count",
            "center_x_px",
            "center_y_px",
            "width_px",
            "depth_px",
        ):
            name = f"team_{team}_{column}"

            if name in report.columns:
                row[f"average_{column}"] = report[name].mean()

        if not report.empty:
            row["frames_estimated_in_possession"] = int(
                (report["estimated_possession_team"] == team).sum()
            )
        else:
            row["frames_estimated_in_possession"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def create_tactical_map(tracks, output_path):
    plt.figure(figsize=(12, 7))

    for team in (TEAM_ARGENTINA, TEAM_ENGLAND):
        xs = []
        ys = []

        for frame_num in range(
            0,
            len(tracks["players"]),
            max(1, MAP_SAMPLE_EVERY),
        ):
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
            plt.scatter(
                xs,
                ys,
                s=8,
                alpha=0.45,
                label=f"Equipo {team}",
            )

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

        cv2.putText(
            debug,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )

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

    cv2.putText(
        frame,
        label,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
        cv2.LINE_AA,
    )

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

def build_field_player_positions(players, view_transformer, frame_num=0):
    result = []

    for track_id, player in players.items():
        role = player.get("role", "field_player")
        team = player.get("team", TEAM_NEUTRAL)

        if role != "field_player":
            continue

        if team not in (TEAM_ARGENTINA, TEAM_ENGLAND):
            continue

        stable_pos = tactical_position(player)
        original_pos = get_player_position(player)

        if stable_pos is None or original_pos is None:
            continue

        world = None

        if view_transformer is not None and getattr(view_transformer, "enabled", False):
            try:
                world = view_transformer.transform_point(
                    original_pos, frame_num=frame_num
                )
            except TypeError:
                # ViewTransformer viejo: sin frame_num
                world = view_transformer.transform_point(original_pos)

        result.append({
            "id": track_id,
            "team": team,
            "x": float(stable_pos[0]),
            "y": float(stable_pos[1]),
            "world": world,
        })

    return result


def nearest_rival_pairs(field_players, max_draw_distance):
    pairs = []
    seen_pairs = set()

    for player in field_players:
        rivals = [
            rival
            for rival in field_players
            if rival["team"] != player["team"]
        ]

        if not rivals:
            continue

        rival = min(
            rivals,
            key=lambda other: math.hypot(
                player["x"] - other["x"],
                player["y"] - other["y"],
            ),
        )

        pair_key = tuple(sorted((
            (player["team"], player["id"]),
            (rival["team"], rival["id"]),
        )))

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)

        dpx = math.hypot(
            player["x"] - rival["x"],
            player["y"] - rival["y"],
        )

        if dpx > max_draw_distance:
            continue

        if player["world"] is not None and rival["world"] is not None:
            distance = float(np.linalg.norm(
                np.asarray(player["world"]) - np.asarray(rival["world"])
            ))
            unit = "m"
        else:
            distance = dpx
            unit = "px"

        pairs.append((dpx, distance, unit, player, rival))

    pairs.sort(key=lambda row: row[0])

    return pairs[:5]


# ============================================================
# VIDEO TÁCTICO
# ============================================================

def create_tactical_video(
    frames,
    tracks,
    team_ball_control,
    output_path,
    eventos_tacticos=None,
    view_transformer=None,
):
    if not frames:
        raise ValueError("No hay fotogramas para generar el video.")

    height, width = frames[0].shape[:2]
    fps = safe_fps(VIDEO_PATH)

    os.makedirs(os.path.dirname(str(output_path)) or ".", exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video: {output_path}")

    # --- Helper interno: transformar punto soportando ambos transformers ---
    def _transform(self, point, frame_num=0):
        if view_transformer is None or not getattr(view_transformer, "enabled", False):
            return None
        if point is None:
            return None
        try:
            return view_transformer.transform_point(point, frame_num=frame_num)
        except TypeError:
            # ViewTransformer viejo: no acepta frame_num
            return view_transformer.transform_point(point)

    previous_positions = {}
    previous_ball_world = None
    previous_ball_frame = None

    total = min(
        len(frames),
        len(tracks["players"]),
        len(tracks["ball"]),
        len(team_ball_control),
    )

    events = eventos_tacticos or []
    defensive_alerts_presentes = bool(events)

    try:
        for frame_num in range(total):
            frame = frames[frame_num].copy()
            players = tracks["players"][frame_num]
            balls = tracks["ball"][frame_num]

            # --- Dibujar jugadores ---
            for track_id, player in players.items():
                draw_player(
                    frame,
                    player,
                    track_id,
                    previous_positions,
                )

            # --- Balón ---
            ball = balls.get(1, {})
            ball_pos = get_ball_position(ball)

            if ball_pos is not None:
                center = tuple(np.round(ball_pos).astype(int))

                cv2.circle(frame, center, 7, YELLOW, 2)

                cv2.putText(
                    frame,
                    "BALON",
                    (center[0] + 8, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    YELLOW,
                    1,
                    cv2.LINE_AA,
                )

            # --- Posesión y análisis por frame ---
            possession = int(team_ball_control[frame_num])

            analysis = analizar_jugada_en_frame(
                players,
                ball_pos,
                possession,
                width,
                height,
            )

            formation_1 = estimate_formation(players, TEAM_ARGENTINA)
            formation_2 = estimate_formation(players, TEAM_ENGLAND)

            compact_1 = estimate_compactness(players, TEAM_ARGENTINA)
            compact_2 = estimate_compactness(players, TEAM_ENGLAND)

            # --- Jugadores de campo con posición estable y mundo ---
            field_players = build_field_player_positions(
                players,
                view_transformer,
                frame_num=frame_num,
            )

            # --- Pares más cercanos entre rivales ---
            pairs = nearest_rival_pairs(
                field_players,
                max_draw_distance=width * 0.105,
            )

            for _, distance, unit, player, rival in pairs:
                p1 = (int(player["x"]), int(player["y"]))
                p2 = (int(rival["x"]), int(rival["y"]))

                cv2.line(frame, p1, p2, YELLOW, 1, cv2.LINE_AA)

                label = (
                    f"E{player['team']}-J{player['id']} vs "
                    f"E{rival['team']}-J{rival['id']}: "
                    f"{distance:.1f} {unit}"
                )

                cx = (p1[0] + p2[0]) // 2
                cy = (p1[1] + p2[1]) // 2

                cv2.putText(
                    frame,
                    label[:95],
                    (max(5, min(width - 400, cx)),
                     max(18, min(height - 5, cy))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    YELLOW,
                    1,
                    cv2.LINE_AA,
                )

            # --- Balón en metros si hay calibración ---
            ball_world = None
            if ball_pos is not None:
                ball_world = _transform(ball_pos, frame_num)

            # --- Velocidad del balón ---
            speed_ms = None
            if ball_world is not None:
                ball_world = np.asarray(ball_world, dtype=float)

                if (
                    previous_ball_world is not None
                    and previous_ball_frame is not None
                ):
                    dt = (frame_num - previous_ball_frame) / fps

                    if dt > 0:
                        speed_ms = float(
                            np.linalg.norm(ball_world - previous_ball_world) / dt
                        )

                previous_ball_world = ball_world
                previous_ball_frame = frame_num
            else:
                previous_ball_world = None
                previous_ball_frame = None

            # --- Distancias de los 3 jugadores más cercanos al balón ---
            if ball_pos is not None:
                near_ball = sorted(
                    (
                        math.hypot(
                            p["x"] - ball_pos[0],
                            p["y"] - ball_pos[1],
                        ),
                        p,
                    )
                    for p in field_players
                )[:3]

                for i, (dpx, player) in enumerate(near_ball):
                    label = (
                        f"E{player['team']}-J{player['id']} balón: "
                        f"{dpx:.0f} px"
                    )

                    if ball_world is not None and player["world"] is not None:
                        distance_m = float(np.linalg.norm(
                            np.asarray(player["world"]) - ball_world
                        ))
                        label = (
                            f"E{player['team']}-J{player['id']} balón: "
                            f"{distance_m:.1f} m"
                        )

                    cv2.putText(
                        frame,
                        label,
                        (max(10, width - 300), 35 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        YELLOW,
                        1,
                        cv2.LINE_AA,
                    )

            # --- Texto de velocidad del balón ---
            if speed_ms is not None:
                speed_text = (
                    f"Velocidad estimada balón: {speed_ms:.1f} m/s "
                    f"({speed_ms * 3.6:.1f} km/h)"
                )
            elif view_transformer is not None and getattr(view_transformer, "enabled", False):
                speed_text = "Velocidad balón: sin datos suficientes"
            else:
                speed_text = "Velocidad en m/s: requiere calibración"

            # --- Texto de posesión ---
            if possession in (TEAM_ARGENTINA, TEAM_ENGLAND):
                possession_text = f"Posesión estimada: Equipo {possession}"
            else:
                possession_text = "Posesión: sin determinar"

            # --- Panel lateral superior izquierdo ---
            panel_lines = [
                ("ANÁLISIS TÁCTICO (ESTIMACIONES)", WHITE),
                (possession_text, TEAM_COLORS.get(possession, WHITE)),
                (f"E1 formación: {formation_1}", TEAM_COLORS[TEAM_ARGENTINA]),
                (f"E2 formación: {formation_2}", TEAM_COLORS[TEAM_ENGLAND]),
                (f"E1 dispersión: {compact_1}", TEAM_COLORS[TEAM_ARGENTINA]),
                (f"E2 dispersión: {compact_2}", TEAM_COLORS[TEAM_ENGLAND]),
                (
                    "Distancias: metros calibrados"
                    if view_transformer is not None
                    and getattr(view_transformer, "enabled", False)
                    else "Distancias: píxeles; falta calibrar",
                    WHITE,
                ),
                (
                    "MODO REVISION: eventos candidatos, no veredictos"
                    if defensive_alerts_presentes
                    else "MODO REVISION: sin eventos en este tramo",
                    WHITE,
                ),
                (speed_text, YELLOW),
            ]

            panel_w = min(720, width - 20)
            panel_h = 210

            cv2.rectangle(
                frame,
                (10, 10),
                (10 + panel_w, min(height - 5, 10 + panel_h)),
                DARK,
                cv2.FILLED,
            )

            y = 34

            for line, color in panel_lines:
                cv2.putText(
                    frame,
                    line[:100],
                    (22, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                y += 22

            time_s = frame_num / fps

            # --- Evento activo (ventana de tiempo del evento) ---
            active_event = next(
                (
                    event for event in events
                    if float(event.get("inicio_seconds", 1e12)) <= time_s
                    <= float(event.get("fin_seconds", -1)) + 0.65
                ),
                None,
            )

            if active_event:
                tipo = str(active_event.get("tipo_evento", "evento")).lower()

                etiquetas = {
                    "intercepcion_defensiva_posible": "CANDIDATO: posible intercepcion",
                    "pase_filtrado_posible": "CANDIDATO: posible pase filtrado",
                    "error_defensivo_posible": "CANDIDATO: receptor con espacio",
                    "pase_completado_posible": "CANDIDATO: pase completado",
                    "receptor_libre_posible": "CANDIDATO: receptor libre",
                }
                title = etiquetas.get(tipo, "CANDIDATO A REVISAR")

                # --- Distancias y velocidad en metros si hay calibración ---
                def _fmt(valor, unidad):
                    if valor in ("", None):
                        return None
                    try:
                        return f"{float(valor):.1f} {unidad}"
                    except (TypeError, ValueError):
                        return None

                dist_pase_px = _fmt(active_event.get("distancia_pase_px"), "px")
                dist_def_px = _fmt(
                    active_event.get("distancia_defensor_receptor_px"), "px"
                )
                vel_px = _fmt(active_event.get("velocidad_pase_px_s"), "px/s")
                vel_m = _fmt(active_event.get("velocidad_pase_m_s"), "m/s")

                lineas = []
                if dist_pase_px:
                    lineas.append(f"Distancia del pase: {dist_pase_px}")
                if dist_def_px:
                    lineas.append(
                        f"Defensor mas cercano al receptor: {dist_def_px}"
                    )
                if vel_m:
                    lineas.append(f"Velocidad del pase: {vel_m}")
                elif vel_px:
                    lineas.append(f"Velocidad del pase: {vel_px}")

                ey = min(max(200, height - 90), height - 70)
                alto_caja = 30 + 22 * (1 + len(lineas))

                cv2.rectangle(
                    frame,
                    (10, ey),
                    (min(width - 10, 950), min(height - 5, ey + alto_caja)),
                    DARK,
                    cv2.FILLED,
                )

                cv2.putText(
                    frame,
                    title[:80],
                    (20, ey + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    YELLOW,
                    2,
                    cv2.LINE_AA,
                )

                y_txt = ey + 44
                for linea in lineas[:4]:
                    cv2.putText(
                        frame,
                        linea[:100],
                        (20, min(height - 8, y_txt)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        WHITE,
                        1,
                        cv2.LINE_AA,
                    )
                    y_txt += 22

            # --- Análisis heurístico por frame (si no hay evento activo) ---
            elif analysis is not None:
                title, detail, color = analysis
                ey = min(220, max(10, height - 70))

                cv2.rectangle(
                    frame,
                    (10, ey),
                    (min(width - 10, 740), min(height - 5, ey + 52)),
                    DARK,
                    cv2.FILLED,
                )

                cv2.putText(
                    frame,
                    title[:70],
                    (20, ey + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    detail[:100],
                    (20, min(height - 8, ey + 43)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    WHITE,
                    1,
                    cv2.LINE_AA,
                )

            # --- Reloj y contador de frames ---
            minutes = int(time_s // 60)
            seconds = int(time_s % 60)

            cv2.putText(
                frame,
                f"{minutes:02d}:{seconds:02d} | "
                f"Frame {frame_num + 1}/{total}",
                (max(10, width - 310), height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                WHITE,
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)

            if (frame_num + 1) % 100 == 0:
                print(f"Video táctico: {frame_num + 1}/{total}")

    finally:
        writer.release()

    print(f"Video táctico generado: {output_path}")

def load_auto_calibrador():
    """
    Carga el AutoCalibrador con Soccana_Keypoint.
    Si falla, cae al modo tramos o al modo simple.
    """
    auto = AutoCalibrador()
    if auto.enabled:
        return auto
    print("[Calibración] AutoCalibrador no disponible; uso tramos/simple.")
    return load_view_transformer_tramos()


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
    print("ANÁLISIS TÁCTICO DE FÚTBOL")
    print("=" * 60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Clip: {CLIP_NAME}")

    fps = safe_fps(VIDEO_PATH)

    # 1. Cargar fotogramas.
    print("\n1/6 - Cargando video...")
    frames = read_video(VIDEO_PATH)

    print(f"Fotogramas: {len(frames)}")
    print(f"FPS: {fps:.2f}")

    # 2. Detección y seguimiento.
    print("\n2/6 - Detectando y siguiendo objetos...")

    tracker = Tracker(MODEL_PATH)

    # Se recalcula el tracking para evitar reutilizar datos viejos.
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=False,
        stub_path=str(TRACKS_STUB),
    )

    if len(tracks["players"]) != len(frames):
        raise RuntimeError(
            "La cantidad de fotogramas no coincide con el tracking: "
            f"{len(frames)} vs {len(tracks['players'])}"
        )

    tracker.add_position_to_tracks(tracks)
    stabilize_track_roles(tracks)

    if len(frames) > 105:
        save_tracking_debug(
            frames[105],
            tracks["players"][105],
            DEBUG_TRACKING_PATH,
        )
        print(f"Diagnóstico guardado: {DEBUG_TRACKING_PATH}")

    print(f"Frames de jugadores: {len(tracks['players'])}")
    print(f"Frames del balón: {len(tracks['ball'])}")

    # 3. Asignación de equipos.
    print("\n3/6 - Separando los equipos...")

    team_assigner = TeamAssigner()

    trained = team_assigner.assign_team_color_multi_frame(
        frames,
        tracks,
        max_frames=60,
    )

    if trained:
        votes = collect_team_votes(
            tracks,
            team_assigner,
            frames,
        )

        resolved = resolve_track_teams(votes)
        apply_resolved_teams(tracks, resolved)

    else:
        # Si no se pueden distinguir los colores, no se inventa equipo.
        apply_resolved_teams(tracks, {})

    # Mantener los roles de árbitros y arqueros; no inferir arqueros
    # simplemente por estar en los extremos del video.
    stabilize_track_roles(tracks)

    goalkeeper_map = identify_goalkeepers(tracks)
    print("Arqueros clasificados por el detector:", goalkeeper_map)

    # 4. Estabilización de cámara y calibración.
    print("\n4/6 - Estabilizando coordenadas y cargando calibración...")

    camera_estimator = CameraMovementEstimator(frames[0])

    camera_movements = camera_estimator.get_camera_movement(
        frames,
        read_from_stub=False,
        stub_path=str(CAMERA_STUB),
    )

    print(f"Desplazamientos de cámara: {len(camera_movements)}")
    camera_estimator.add_adjust_positions_to_tracks(
        tracks,
        camera_movements,
    )

    # ------------------------------------------------------------------
    # NUEVO: carga del calibrador automático con Soccana_Keypoint
    # ------------------------------------------------------------------
    view_transformer = load_auto_calibrador()

    # NUEVO: calibrar los frames ANTES de transformar los tracks.
    # Esto recorre el video con YOLO y guarda una homografía por frame
    # (o cada N frames). Si no se detecta el área, cae al modo tramos.
    if hasattr(view_transformer, "calibrar_frames"):
        view_transformer.calibrar_frames(frames)

    view_transformer.add_transformed_position_to_tracks(tracks)
    # ------------------------------------------------------------------

    referee_tracks = set()

    for frame_players in tracks["players"]:
        for track_id, player in frame_players.items():
            if player.get("role") == "referee":
                referee_tracks.add(track_id)

    print("Tracks de árbitros:", referee_tracks)

    # 5. Métricas y análisis defensivo.
    print("\n5/6 - Calculando métricas...")

    possession = estimate_possession(tracks)

    defensive_alerts = detectar_alertas_defensivas(
        tracks=tracks,
        fps=fps,
        salida_csv=OUTPUT_DIR / f"defensa_{CLIP_NAME}.csv",
        tiempos_goles=GOAL_TIMES_SECONDS,
        equipo_en_posesion=possession,
        view_transformer=view_transformer,
    )

    if defensive_alerts is None:
        defensive_alerts = []

    report = build_tactical_report(
        tracks,
        possession,
        fps=fps,
    )

    report.to_csv(REPORT_PATH, index=False)
    create_summary(report).to_csv(SUMMARY_PATH, index=False)
    create_tactical_map(tracks, MAP_PATH)

    create_tactical_video(
        frames=frames,
        tracks=tracks,
        team_ball_control=possession,
        output_path=TACTICAL_VIDEO_PATH,
        eventos_tacticos=defensive_alerts,
        view_transformer=view_transformer,
    )

    # 6. Resumen.
    print("\n6/6 - Resultados")
    print(f"Video táctico: {TACTICAL_VIDEO_PATH}")
    print(f"Informe:       {REPORT_PATH}")
    print(f"Resumen:       {SUMMARY_PATH}")
    print(f"Mapa:          {MAP_PATH}")
    print(f"Diagnóstico:   {DEBUG_TRACKING_PATH}")

    print("\nPrimeras filas del informe:")
    print(report.head(10).to_string(index=False))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    main()