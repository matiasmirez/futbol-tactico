# ============================================================
# FUTBOL.PY - Archivo autocontenido (CORREGIDO)
# Todo el pipeline de análisis táctico en un solo archivo.
# ============================================================

import os
import sys
import pickle
from pathlib import Path
from collections import Counter, defaultdict, deque
from defense import detectar_alertas_defensivas
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from ultralytics import YOLO
import supervision as sv


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# --- RUTA DEL VIDEO -------------------------------------------------
# Por defecto, un clip corto (30 s) para evitar problemas de RAM.
# Puedes sobreescribirla por linea de comandos:
#     python futbol.py "C:\ruta\al\clip.mp4"
DEFAULT_VIDEO_PATH = Path(
    r"C:\Users\Matias\Downloads\chivas_penal.mp4"
)

if len(sys.argv) > 1:
    VIDEO_PATH = Path(sys.argv[1])
else:
    VIDEO_PATH = DEFAULT_VIDEO_PATH

# Nombre unico por clip para no pisar stubs/salidas entre clips
CLIP_NAME = VIDEO_PATH.stem  # p.ej. "clip_000"

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

GOAL_TIMES_SECONDS = []

MAP_SAMPLE_EVERY = 5

TEAM_ARGENTINA = 1
TEAM_ENGLAND = 2
TEAM_NEUTRAL = 0

TEAM_COLORS = {
    TEAM_ARGENTINA: (255, 120, 0),   # Azul, BGR
    TEAM_ENGLAND: (0, 80, 255),      # Rojo, BGR
    TEAM_NEUTRAL: (180, 180, 180),   # Gris
}

GOALKEEPER_COLOR = (0, 220, 255)     # Amarillo
REFEREE_COLOR = (0, 255, 0)          # Verde

WHITE = (255, 255, 255)
GREEN = (70, 220, 90)
YELLOW = (0, 220, 255)
RED = (0, 0, 255)
DARK = (25, 25, 25)

FORMATION_AXIS = "image_y"
EXPECTED_TEAM_PLAYERS = 11
MIN_TEAM_VOTES = 3
MIN_TEAM_VOTE_RATIO = 0.60
LINE_GAP_FACTOR = 2.2
MAX_FORMATION_LINES = 4
MIN_PLAYER_HEIGHT_PX = 18
EDGE_MARGIN_PX = 3


# ============================================================
# UTILS
# ============================================================

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    return bbox[2] - bbox[0]


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]


# ============================================================
# VIDEO UTILS
# ============================================================

def read_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"El video no contiene fotogramas: {video_path}")

    return frames


def save_video(output_video_frames, output_video_path, fps=24):
    if not output_video_frames:
        raise ValueError("No hay fotogramas para guardar.")

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    height, width = output_video_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        raise RuntimeError(
            f"No se pudo crear el video de salida: {output_video_path}"
        )

    try:
        for frame in output_video_frames:
            out.write(frame)
    finally:
        out.release()


# ============================================================
# TRACKER
# ============================================================

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]

                    if object_name == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)

                    tracks[object_name][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions,
            columns=["x1", "y1", "x2", "y2"],
        )

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {"bbox": x}}
            for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i + batch_size],
                conf=0.1,
                verbose=False,
            )
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # ---------- Cargar stub SOLO si coincide en longitud ----------
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            try:
                with open(stub_path, "rb") as f:
                    cached = pickle.load(f)
                cached_len = len(cached.get("players", []))
                if cached_len == len(frames):
                    print(f"[Tracker] Stub valido ({cached_len} frames).")
                    return cached
                else:
                    print(
                        f"[Tracker] Stub con {cached_len} frames, "
                        f"se esperaban {len(frames)}. Recalculando..."
                    )
            except Exception as e:
                print(f"[Tracker] Error leyendo stub: {e}. Recalculando...")

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names = {int(cid): name for cid, name in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id_value = frame_detection[3]
                track_id_value = frame_detection[4]

                if class_id_value is None or track_id_value is None:
                    continue

                class_id = int(class_id_value)
                track_id = int(track_id_value)
                class_name = cls_names.get(class_id)

                if class_name == "player":
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "role": "field_player",
                        "class_name": "player",
                    }

                elif class_name == "goalkeeper":
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "role": "goalkeeper",
                        "class_name": "goalkeeper",
                    }

                elif class_name == "referee":
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "role": "referee",
                        "class_name": "referee",
                    }

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id_value = frame_detection[3]

                if class_id_value is None:
                    continue

                class_id = int(class_id_value)
                class_name = cls_names.get(class_id)

                if class_name == "ball":
                    tracks["ball"][frame_num][1] = {
                        "bbox": bbox,
                        "class_name": "ball",
                    }

        if stub_path is not None:
            stub_directory = os.path.dirname(stub_path)
            if stub_directory:
                os.makedirs(stub_directory, exist_ok=True)

            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        control_hasta_ahora = team_ball_control[:frame_num + 1]

        team_1_num_frames = np.count_nonzero(control_hasta_ahora == 1)
        team_2_num_frames = np.count_nonzero(control_hasta_ahora == 2)

        total = team_1_num_frames + team_2_num_frames

        x1, y1 = 1350, 850
        x2, y2 = 1900, 970

        height, width = frame.shape[:2]
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x1 < x2 and y1 < y2:
            roi = frame[y1:y2, x1:x2]
            overlay = np.full_like(roi, (255, 255, 255))
            cv2.addWeighted(overlay, 0.4, roi, 0.6, 0, roi)

        if total > 0:
            team_1 = team_1_num_frames / total * 100
            team_2 = team_2_num_frames / total * 100
        else:
            team_1 = 0.0
            team_2 = 0.0

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )

        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                role = player.get("role", "field_player")

                if role == "goalkeeper":
                    color = GOALKEEPER_COLOR
                elif role == "referee":
                    color = REFEREE_COLOR
                else:
                    color = player.get("team_color", (0, 0, 255))

                frame = self.draw_ellipse(
                    frame,
                    player["bbox"],
                    color,
                    track_id,
                )

                if player.get("has_ball", False):
                    frame = self.draw_traingle(
                        frame,
                        player["bbox"],
                        (0, 0, 255),
                    )

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(
                    frame,
                    referee["bbox"],
                    REFEREE_COLOR,
                )

            for _, ball in ball_dict.items():
                frame = self.draw_traingle(
                    frame,
                    ball["bbox"],
                    (0, 255, 0),
                )

            frame = self.draw_team_ball_control(
                frame,
                frame_num,
                team_ball_control,
            )

            output_video_frames.append(frame)

        return output_video_frames


# ============================================================
# TEAM ASSIGNER
# ============================================================

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_history = {}
        self.kmeans = None
        self.history_size = 7

    def get_player_color(self, frame, bbox):
        if bbox is None or frame is None:
            return np.zeros(3, dtype=np.float32)

        x1, y1, x2, y2 = map(int, bbox)

        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return np.zeros(3, dtype=np.float32)

        player = frame[y1:y2, x1:x2]

        height, width = player.shape[:2]

        if height < 20 or width < 10:
            return np.zeros(3, dtype=np.float32)

        torso_y1 = int(height * 0.20)
        torso_y2 = int(height * 0.50)
        torso_x1 = int(width * 0.22)
        torso_x2 = int(width * 0.78)

        torso = player[torso_y1:torso_y2, torso_x1:torso_x2]

        if torso.size == 0:
            return np.zeros(3, dtype=np.float32)

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        pixels = hsv.reshape(-1, 3).astype(np.float32)

        brightness = pixels[:, 2]

        valid_mask = brightness > 35
        valid = pixels[valid_mask]

        if len(valid) < 5:
            return np.zeros(3, dtype=np.float32)

        hue = np.median(valid[:, 0]) / 180.0
        sat = np.median(valid[:, 1]) / 255.0
        val = np.median(valid[:, 2]) / 255.0

        weighted_hue = hue * sat

        return np.array([weighted_hue, sat, val], dtype=np.float32)

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, image.shape[-1])

        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=10,
            random_state=42,
        )
        kmeans.fit(image_2d)

        return kmeans

    def _reset_player_history(self):
        self.player_team_dict.clear()
        self.player_team_history.clear()

    def assign_team_color(self, frame, player_detections):
        player_colors = []

        for _, detection in player_detections.items():
            role = detection.get("role", "field_player")
            if role in ("goalkeeper", "referee"):
                continue

            bbox = detection.get("bbox")
            if bbox is None:
                continue

            color = self.get_player_color(frame, bbox)
            if np.linalg.norm(color) > 1e-6:
                player_colors.append(color)

        if len(player_colors) < 2:
            print("[TeamAssigner] No hay suficientes colores válidos.")
            return False

        player_colors = np.asarray(player_colors, dtype=np.float32)

        if np.unique(np.round(player_colors, decimals=3), axis=0).shape[0] < 2:
            print("[TeamAssigner] Colores demasiado similares.")
            return False

        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=10,
            random_state=42,
        )

        labels = kmeans.fit_predict(player_colors)

        if len(np.unique(labels)) < 2:
            print("[TeamAssigner] KMeans no pudo separar equipos.")
            return False

        self.kmeans = kmeans
        self.team_colors = {
            1: kmeans.cluster_centers_[0].copy(),
            2: kmeans.cluster_centers_[1].copy(),
        }

        self._reset_player_history()

        print(f"[TeamAssigner] Entrenado con {len(player_colors)} jugadores de un frame.")
        print("Centro equipo 1:", self.team_colors[1])
        print("Centro equipo 2:", self.team_colors[2])
        return True

    def assign_team_color_multi_frame(self, frames, tracks, max_frames=50):
        all_colors = []
        frames_used = 0

        total_frames = min(len(frames), len(tracks["players"]), max_frames)

        for frame_num in range(total_frames):
            players = tracks["players"][frame_num]
            if not players:
                continue

            frame_colors = []
            for _, player in players.items():
                role = player.get("role", "field_player")
                if role in ("goalkeeper", "referee"):
                    continue

                bbox = player.get("bbox")
                if bbox is None:
                    continue

                color = self.get_player_color(frames[frame_num], bbox)
                if np.linalg.norm(color) > 1e-6:
                    frame_colors.append(color)

            if len(frame_colors) >= 4:
                all_colors.extend(frame_colors)
                frames_used += 1

        if len(all_colors) < 4:
            print("[TeamAssigner] Pocas muestras en varios frames.")
            return False

        all_colors = np.asarray(all_colors, dtype=np.float32)

        unique_colors = np.unique(np.round(all_colors, decimals=3), axis=0)
        if len(unique_colors) < 2:
            print("[TeamAssigner] Poca diversidad de colores.")
            return False

        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=10,
            random_state=42,
        )

        labels = kmeans.fit_predict(all_colors)

        if len(np.unique(labels)) < 2:
            print("[TeamAssigner] No fue posible separar equipos.")
            return False

        self.kmeans = kmeans
        self.team_colors = {
            1: kmeans.cluster_centers_[0].copy(),
            2: kmeans.cluster_centers_[1].copy(),
        }

        self._reset_player_history()

        print(f"[TeamAssigner] Entrenado con {len(all_colors)} muestras de {frames_used} frames.")
        return True

    def get_player_team(self, frame, player_bbox, player_id, role="field_player"):
        if role in ("goalkeeper", "referee"):
            return self.player_team_dict.get(player_id, 0)

        if self.kmeans is None:
            return self.player_team_dict.get(player_id, 1)

        player_color = self.get_player_color(frame, player_bbox)

        if np.linalg.norm(player_color) <= 1e-6:
            return self.player_team_dict.get(player_id, 1)

        prediction = int(
            self.kmeans.predict(player_color.reshape(1, -1))[0]
        ) + 1

        if player_id not in self.player_team_history:
            self.player_team_history[player_id] = deque(maxlen=self.history_size)

        history = self.player_team_history[player_id]
        history.append(prediction)

        counts = Counter(history)
        max_votes = max(counts.values())

        candidates = [team for team, votes in counts.items() if votes == max_votes]

        if len(candidates) == 1:
            team_id = candidates[0]
        else:
            team_id = prediction

        self.player_team_dict[player_id] = team_id
        return team_id


# ============================================================
# CAMERA MOVEMENT ESTIMATOR
# ============================================================

class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                # --- Defensa contra longitudes inconsistentes ---
                if frame_num >= len(camera_movement_per_frame):
                    continue
                for track_id, track_info in track.items():
                    position = track_info.get("position")
                    if position is None:
                        continue
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1],
                    )
                    tracks[object_name][frame_num][track_id]["position_adjusted"] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            try:
                with open(stub_path, "rb") as f:
                    camera_movement = pickle.load(f)
                return camera_movement
            except (pickle.UnpicklingError, EOFError, FileNotFoundError):
                pass

        camera_movement = []

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(
            old_gray,
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        for frame in frames:
            new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if old_features is not None and len(old_features) > 0:
                new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray,
                    new_gray,
                    old_features,
                    None
                )

                if new_features is not None and status is not None:
                    good_old = old_features[status.flatten() == 1]
                    good_new = new_features[status.flatten() == 1]

                    if len(good_old) > 0 and len(good_new) > 0:
                        movement = good_new - good_old
                        avg_movement = movement.mean(axis=0).flatten()
                        camera_movement.append(avg_movement)
                    else:
                        camera_movement.append([0, 0])

                    old_features = new_features
                else:
                    camera_movement.append([0, 0])
                    old_features = cv2.goodFeaturesToTrack(
                        new_gray,
                        maxCorners=100,
                        qualityLevel=0.3,
                        minDistance=7,
                        blockSize=7
                    )
            else:
                camera_movement.append([0, 0])
                old_features = cv2.goodFeaturesToTrack(
                    new_gray,
                    maxCorners=100,
                    qualityLevel=0.3,
                    minDistance=7,
                    blockSize=7
                )

            old_gray = new_gray.copy()

        if stub_path is not None:
            try:
                with open(stub_path, "wb") as f:
                    pickle.dump(camera_movement, f)
            except OSError as e:
                print(f"No se pudo guardar el stub de movimiento de cámara: {e}")

        return camera_movement
# ============================================================
# VIEW TRANSFORMER
# ============================================================

class ViewTransformer:
    """
    Transforma coordenadas de imagen a coordenadas de cancha
    usando una homografía (perspectiva).
    """

    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915],
        ])

        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width],
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices,
            self.target_vertices,
        )

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))

        is_inside = cv2.pointPolygonTest(
            self.pixel_vertices,
            p,
            False,
        ) >= 0

        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)

        transformed_point = cv2.perspectiveTransform(
            reshaped_point,
            self.perspective_transformer,
        )

        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():

                    position = track_info.get(
                        "position_adjusted",
                        track_info.get("position"),
                    )

                    if position is None:
                        tracks[object_name][frame_num][track_id][
                            "position_transformed"
                        ] = None
                        continue

                    position = np.array(position, dtype=np.float32)

                    position_transformed = self.transform_point(position)

                    if position_transformed is not None:
                        position_transformed = (
                            position_transformed.squeeze().tolist()
                        )

                    tracks[object_name][frame_num][track_id][
                        "position_transformed"
                    ] = position_transformed


# ============================================================
# POSICIONES
# ============================================================

def get_player_position(player):
    bbox = player.get("bbox")
    if bbox is None or len(bbox) < 4:
        return None

    x1, y1, x2, y2 = bbox
    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return None

    return ((x1 + x2) / 2.0, y2)


def get_ball_position(ball):
    bbox = ball.get("bbox")
    if bbox is not None and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    position = ball.get("position")
    if position is not None and len(position) >= 2:
        return tuple(position[:2])

    return None


def bbox_is_valid(bbox):
    if bbox is None or len(bbox) < 4:
        return False
    x1, y1, x2, y2 = bbox
    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return False
    return x2 > x1 and y2 > y1


def player_bbox_size(player):
    bbox = player.get("bbox")
    if not bbox_is_valid(bbox):
        return 0, 0
    x1, y1, x2, y2 = bbox
    return float(x2 - x1), float(y2 - y1)


def tactical_position(player):
    for key in ("position_transformed", "position_adjusted", "position"):
        value = player.get(key)
        if value is not None and len(value) >= 2:
            value = np.asarray(value[:2], dtype=float)
            if np.all(np.isfinite(value)):
                return value
    return None


def is_valid_player_detection(player):
    bbox = player.get("bbox")
    if not bbox_is_valid(bbox):
        return False

    width, height = player_bbox_size(player)
    if height < MIN_PLAYER_HEIGHT_PX:
        return False
    if width <= 0:
        return False
    return True


def is_neutral_team(team):
    return team not in (TEAM_ARGENTINA, TEAM_ENGLAND)


# ============================================================
# ASIGNACION ESTABLE DE EQUIPO
# ============================================================

def collect_team_votes(tracks, team_assigner, frames):
    votes = defaultdict(list)
    total_frames = min(len(frames), len(tracks["players"]))

    for frame_num in range(total_frames):
        player_tracks = tracks["players"][frame_num]

        for track_id, player in player_tracks.items():
            if not is_valid_player_detection(player):
                continue

            role = player.get("role", "field_player")

            if role in ("goalkeeper", "referee"):
                votes[track_id].append(TEAM_NEUTRAL)
                continue

            try:
                predicted_team = team_assigner.get_player_team(
                    frames[frame_num],
                    player["bbox"],
                    track_id,
                    role=role,
                )
            except Exception:
                predicted_team = TEAM_NEUTRAL

            if predicted_team not in (TEAM_ARGENTINA, TEAM_ENGLAND):
                predicted_team = TEAM_NEUTRAL

            votes[track_id].append(predicted_team)

    return votes


def resolve_track_teams(team_votes):
    resolved = {}

    for track_id, votes in team_votes.items():
        valid_votes = [v for v in votes if v in (TEAM_ARGENTINA, TEAM_ENGLAND)]

        if len(valid_votes) < MIN_TEAM_VOTES:
            resolved[track_id] = TEAM_NEUTRAL
            continue

        counts = Counter(valid_votes)
        team, count = counts.most_common(1)[0]
        ratio = count / len(valid_votes)

        if ratio < MIN_TEAM_VOTE_RATIO:
            resolved[track_id] = TEAM_NEUTRAL
        else:
            resolved[track_id] = team

    return resolved


def apply_resolved_teams(tracks, resolved):
    for frame_players in tracks["players"]:
        for track_id, player in frame_players.items():
            team = resolved.get(track_id, TEAM_NEUTRAL)
            player["team"] = int(team)

            role = player.get("role", "field_player")

            if role == "referee":
                player["team"] = TEAM_NEUTRAL
                player["team_color"] = REFEREE_COLOR
                continue

            if role == "goalkeeper":
                player["team_color"] = GOALKEEPER_COLOR
                continue

            player["team_color"] = TEAM_COLORS.get(
                team, TEAM_COLORS[TEAM_NEUTRAL]
            )


# ============================================================
# ARQUEROS
# ============================================================

def identify_goalkeepers(tracks):
    result = {TEAM_ARGENTINA: None, TEAM_ENGLAND: None}

    latest_positions = {}

    for frame_players in tracks["players"]:
        for track_id, player in frame_players.items():
            if player.get("role") == "referee":
                continue

            team = player.get("team")
            if team not in (TEAM_ARGENTINA, TEAM_ENGLAND):
                continue

            position = get_player_position(player)
            if position is None:
                continue

            latest_positions.setdefault(track_id, []).append(position)

    team_tracks = defaultdict(list)

    for track_id, positions in latest_positions.items():
        team = None
        for frame_players in tracks["players"]:
            if track_id in frame_players:
                if frame_players[track_id].get("role") == "referee":
                    team = None
                    break
                team = frame_players[track_id].get("team")
                if team in (TEAM_ARGENTINA, TEAM_ENGLAND):
                    break

        if team in (TEAM_ARGENTINA, TEAM_ENGLAND):
            median_position = np.median(np.asarray(positions), axis=0)
            team_tracks[team].append((track_id, median_position, len(positions)))

    for team in (TEAM_ARGENTINA, TEAM_ENGLAND):
        candidates = team_tracks.get(team, [])
        if len(candidates) < 4:
            continue

        candidates.sort(key=lambda item: item[1][0])

        left_candidate = candidates[0]
        right_candidate = candidates[-1]

        if left_candidate[2] > right_candidate[2] * 1.25:
            chosen = left_candidate
        elif right_candidate[2] > left_candidate[2] * 1.25:
            chosen = right_candidate
        else:
            chosen = None

        if chosen is not None:
            result[team] = chosen[0]

    for frame_players in tracks["players"]:
        for track_id, player in frame_players.items():
            if player.get("role") == "referee":
                continue

            player["role"] = "field_player"

            if track_id == result[TEAM_ARGENTINA]:
                player["role"] = "goalkeeper"
                player["team"] = TEAM_ARGENTINA
                player["team_color"] = GOALKEEPER_COLOR

            elif track_id == result[TEAM_ENGLAND]:
                player["role"] = "goalkeeper"
                player["team"] = TEAM_ENGLAND
                player["team_color"] = GOALKEEPER_COLOR

    return result


# ============================================================
# POSESION ESTIMADA
# ============================================================

def estimate_possession(tracks):
    possession_by_frame = []
    last_team = TEAM_NEUTRAL

    players_per_frame = tracks["players"]
    balls_per_frame = tracks["ball"]

    total_frames = min(len(players_per_frame), len(balls_per_frame))

    for frame_num in range(total_frames):
        players = players_per_frame[frame_num]
        ball_data = balls_per_frame[frame_num].get(1, {})
        ball_position = get_ball_position(ball_data)

        if ball_position is not None:
            nearest_team = None
            nearest_distance = float("inf")

            for player in players.values():
                team = player.get("team")
                if team not in (TEAM_ARGENTINA, TEAM_ENGLAND):
                    continue
                if player.get("role") == "referee":
                    continue

                player_position = get_player_position(player)
                if player_position is None:
                    continue

                distance = np.hypot(
                    player_position[0] - ball_position[0],
                    player_position[1] - ball_position[1],
                )

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_team = team

            if nearest_team is not None:
                last_team = nearest_team

        possession_by_frame.append(last_team)

    while len(possession_by_frame) < len(players_per_frame):
        possession_by_frame.append(last_team)

    return np.asarray(possession_by_frame, dtype=int)


# ============================================================
# FORMACION
# ============================================================

def formation_coordinate(player):
    position = get_player_position(player)
    if position is None:
        return None

    if FORMATION_AXIS == "image_y":
        return float(position[1])

    tactical = tactical_position(player)
    if tactical is None:
        return float(position[1])

    return float(tactical[1])


def split_into_formation_lines(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return [values]

    values = np.sort(values)
    gaps = np.diff(values)

    if len(gaps) == 0:
        return [values]

    positive_gaps = gaps[gaps > 0]
    if len(positive_gaps) == 0:
        return [values]

    median_gap = float(np.median(positive_gaps))
    if median_gap <= 0:
        return [values]

    threshold = max(median_gap * LINE_GAP_FACTOR, median_gap + 8.0)

    split_indices = np.where(gaps > threshold)[0]

    if len(split_indices) > MAX_FORMATION_LINES - 1:
        largest_gap_indices = np.argsort(gaps[split_indices])[
            -(MAX_FORMATION_LINES - 1):
        ]
        split_indices = np.sort(split_indices[largest_gap_indices])

    groups = np.split(values, split_indices + 1)
    return [group for group in groups if len(group) > 0]


def estimate_formation(players, team):
    values = []
    goalkeeper_present = False

    for player in players.values():
        if player.get("team") != team:
            continue
        if player.get("role") == "referee":
            continue
        if player.get("role") == "goalkeeper":
            goalkeeper_present = True
            continue
        if not is_valid_player_detection(player):
            continue

        coordinate = formation_coordinate(player)
        if coordinate is None:
            continue

        values.append(coordinate)

    if len(values) < 5:
        return f"Jugadores insuficientes ({len(values)})"

    groups = split_into_formation_lines(values)
    counts = [len(group) for group in groups]
    formation = "-".join(map(str, counts))
    role_suffix = " + GK" if goalkeeper_present else ""

    return f"{formation}{role_suffix} (n={len(values)})"


# ============================================================
# METRICAS
# ============================================================

def estimate_compactness(players, team):
    positions = []

    for player in players.values():
        if player.get("team") != team:
            continue
        if player.get("role") in ("referee", "goalkeeper"):
            continue

        position = tactical_position(player)
        if position is not None and np.all(np.isfinite(position)):
            positions.append(position)

    if len(positions) < 4:
        return "Dispersion: sin datos"

    points = np.asarray(positions)
    spread = float(np.mean(np.std(points, axis=0)))
    return f"Dispersion media: {spread:.1f} unidades"


def calculate_team_metrics(players):
    result = {}

    for team in (TEAM_ARGENTINA, TEAM_ENGLAND):
        positions = []

        for player in players.values():
            if player.get("team") != team:
                continue
            if player.get("role") in ("referee", "goalkeeper"):
                continue

            position = get_player_position(player)
            if position is not None:
                positions.append(position)

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


def build_tactical_report(tracks, team_ball_control):
    rows = []
    total_frames = len(tracks["players"])

    for frame_num in range(total_frames):
        players = tracks["players"][frame_num]
        metrics = calculate_team_metrics(players)

        row = {
            "frame": frame_num,
            "time_seconds": frame_num / 25.0,
            "estimated_possession_team": int(team_ball_control[frame_num]),
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

        for column in ("player_count", "center_x_px", "center_y_px", "width_px", "depth_px"):
            name = f"team_{team}_{column}"
            if name in report.columns:
                row[f"average_{column}"] = report[name].mean()

        row["frames_estimated_in_possession"] = int(
            (report["estimated_possession_team"] == team).sum()
        )

        rows.append(row)

    return pd.DataFrame(rows)


def create_tactical_map(tracks, output_path):
    plt.figure(figsize=(12, 7))

    for team in (TEAM_ARGENTINA, TEAM_ENGLAND):
        x_values = []
        y_values = []

        for frame_num in range(0, len(tracks["players"]), MAP_SAMPLE_EVERY):
            for player in tracks["players"][frame_num].values():
                if player.get("team") != team:
                    continue
                if player.get("role") in ("referee", "goalkeeper"):
                    continue

                position = get_player_position(player)
                if position is None:
                    continue

                x_values.append(position[0])
                y_values.append(position[1])

        if x_values:
            plt.scatter(x_values, y_values, s=8, alpha=0.45, label=f"Equipo {team}")

    plt.title("Distribucion espacial de jugadores")
    plt.xlabel("Coordenada horizontal de imagen (px)")
    plt.ylabel("Coordenada vertical de imagen (px)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# DEBUG
# ============================================================

def save_tracking_debug(frame, players, output_path):
    debug_frame = frame.copy()

    for track_id, player in players.items():
        bbox = player.get("bbox")
        if not bbox_is_valid(bbox):
            continue

        x1, y1, x2, y2 = map(int, bbox)

        team = player.get("team", TEAM_NEUTRAL)
        role = player.get("role", "unknown")

        if role == "goalkeeper":
            color = GOALKEEPER_COLOR
        elif role == "referee":
            color = REFEREE_COLOR
        else:
            color = TEAM_COLORS.get(team, TEAM_COLORS[TEAM_NEUTRAL])

        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {track_id} E{team} {role}"
        cv2.putText(
            debug_frame,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )

    cv2.imwrite(str(output_path), debug_frame)


# ============================================================
# DIBUJO
# ============================================================

def draw_player(frame, player, track_id, previous_positions):
    bbox = player.get("bbox")
    if not bbox_is_valid(bbox):
        return

    x1, y1, x2, y2 = map(int, bbox)

    team = player.get("team", TEAM_NEUTRAL)
    role = player.get("role", "field_player")

    if role == "goalkeeper":
        color = GOALKEEPER_COLOR
    elif role == "referee":
        color = REFEREE_COLOR
    else:
        color = TEAM_COLORS.get(team, TEAM_COLORS[TEAM_NEUTRAL])

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if role == "goalkeeper":
        label = f"J{track_id} | E{team} | GK"
    elif role == "referee":
        label = f"J{track_id} | ARBITRO"
    elif team in (TEAM_ARGENTINA, TEAM_ENGLAND):
        label = f"J{track_id} | E{team}"
    else:
        label = f"J{track_id} | NEUTRAL"

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
        cv2.LINE_AA,
    )

    position = get_player_position(player)
    if position is None:
        return

    foot = (int(position[0]), int(position[1]))
    cv2.circle(frame, foot, 4, color, -1)

    old_point = previous_positions.get(track_id)
    if old_point is not None:
        cv2.line(frame, old_point, foot, color, 2)

    previous_positions[track_id] = foot


# ============================================================
# VIDEO TACTICO
# ============================================================


def create_tactical_video(frames, tracks, team_ball_control, output_path):
    if not frames:
        raise ValueError("No hay fotogramas.")

    height, width = frames[0].shape[:2]

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    output_path = str(output_path)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear: {output_path}")

    previous_positions = {}

    total_frames = min(
        len(frames),
        len(tracks["players"]),
        len(tracks["ball"]),
        len(team_ball_control),
    )

    try:
        for frame_num in range(total_frames):
            frame = frames[frame_num].copy()
            players = tracks["players"][frame_num]
            balls = tracks["ball"][frame_num]

            # ------------------------------------------------
            # 1. DIBUJAR JUGADORES
            # ------------------------------------------------
            for track_id, player in players.items():
                draw_player(frame, player, track_id, previous_positions)

            # ------------------------------------------------
            # 2. POSICION DE LA PELOTA
            # ------------------------------------------------
            ball = balls.get(1, {})
            ball_position = get_ball_position(ball)

            if ball_position is not None:
                ball_center = tuple(map(int, ball_position))

                cv2.circle(frame, ball_center, 9, YELLOW, 3)

                cv2.putText(
                    frame,
                    "PELOTA",
                    (ball_center[0] + 10, ball_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    YELLOW,
                    2,
                    cv2.LINE_AA,
                )

            possession = int(team_ball_control[frame_num])

            # ------------------------------------------------
            # 3. ANALISIS DE LA JUGADA
            # ------------------------------------------------
            analisis_jugada = analizar_jugada_en_frame(
                players,
                ball_position,
                possession,
                width,
                height,
            )

            if possession in (TEAM_ARGENTINA, TEAM_ENGLAND):
                possession_text = f"Posesion estimada: Equipo {possession}"
                possession_color = TEAM_COLORS[possession]
            else:
                possession_text = "Posesion: sin determinar"
                possession_color = WHITE

            # ------------------------------------------------
            # 4. FORMACIONES Y COMPACTACION
            # ------------------------------------------------
            formation_1 = estimate_formation(players, TEAM_ARGENTINA)
            formation_2 = estimate_formation(players, TEAM_ENGLAND)

            compactness_1 = estimate_compactness(players, TEAM_ARGENTINA)
            compactness_2 = estimate_compactness(players, TEAM_ENGLAND)

            # ------------------------------------------------
            # 5. DISTANCIA DE LOS RIVALES A LA PELOTA
            # ------------------------------------------------
            pressure_distances = []

            if (
                ball_position is not None
                and possession in (TEAM_ARGENTINA, TEAM_ENGLAND)
            ):
                for player in players.values():
                    team = player.get("team")

                    if team not in (TEAM_ARGENTINA, TEAM_ENGLAND):
                        continue

                    if team == possession:
                        continue

                    if player.get("role") == "referee":
                        continue

                    pixel_position = get_player_position(player)

                    if pixel_position is None:
                        continue

                    distance = float(
                        np.hypot(
                            pixel_position[0] - ball_position[0],
                            pixel_position[1] - ball_position[1],
                        )
                    )

                    pressure_distances.append(distance)

            nearby_opponents = sum(
                d < width * 0.06 for d in pressure_distances
            )

            if (
                ball_position is None
                or possession not in (TEAM_ARGENTINA, TEAM_ENGLAND)
            ):
                pressure_text = "Presion: sin datos suficientes"
                pressure_color = WHITE
            elif nearby_opponents >= 3:
                pressure_text = "Presion cercana intensa probable"
                pressure_color = RED
            elif nearby_opponents == 2:
                pressure_text = "Varios rivales cerca del balon"
                pressure_color = YELLOW
            elif nearby_opponents == 1:
                pressure_text = "Un rival cerca del balon"
                pressure_color = GREEN
            else:
                pressure_text = "Poca presion cercana"
                pressure_color = WHITE

            # ------------------------------------------------
            # 6. PANEL TACTICO
            # ------------------------------------------------
            panel_width = min(600, width - 20)
            panel_height = 170

            cv2.rectangle(
                frame,
                (10, 10),
                (10 + panel_width, 10 + panel_height),
                DARK,
                -1,
            )

            panel_lines = [
                ("ANALISIS TACTICO", WHITE),
                (possession_text, possession_color),
                (f"E1: {formation_1}", TEAM_COLORS[TEAM_ARGENTINA]),
                (f"E2: {formation_2}", TEAM_COLORS[TEAM_ENGLAND]),
                (pressure_text, pressure_color),
                (f"E1 {compactness_1}", TEAM_COLORS[TEAM_ARGENTINA]),
                (f"E2 {compactness_2}", TEAM_COLORS[TEAM_ENGLAND]),
                ("Distancias expresadas en pixeles (px)", WHITE),
            ]

            y_text = 36

            for line_text, color in panel_lines:
                cv2.putText(
                    frame,
                    line_text[:100],
                    (22, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    1 if line_text.startswith("E") else 2,
                    cv2.LINE_AA,
                )
                y_text += 22

            # ------------------------------------------------
            # 7. DISTANCIAS ENTRE JUGADORES RIVALES
            # ------------------------------------------------
            posiciones_jugadores = []

            for track_id, player in players.items():
                if player.get("role") == "referee":
                    continue

                team = player.get("team")

                if team not in (TEAM_ARGENTINA, TEAM_ENGLAND):
                    continue

                position = get_player_position(player)

                if position is None:
                    continue

                posiciones_jugadores.append({
                    "id": track_id,
                    "team": team,
                    "x": float(position[0]),
                    "y": float(position[1]),
                })

            pares_mostrados = set()
            distancias_jugadores = []

            for jugador in posiciones_jugadores:
                rivales = [
                    rival
                    for rival in posiciones_jugadores
                    if rival["team"] != jugador["team"]
                ]

                if not rivales:
                    continue

                rival_cercano = min(
                    rivales,
                    key=lambda rival: np.hypot(
                        jugador["x"] - rival["x"],
                        jugador["y"] - rival["y"],
                    ),
                )

                distancia = float(
                    np.hypot(
                        jugador["x"] - rival_cercano["x"],
                        jugador["y"] - rival_cercano["y"],
                    )
                )

                par = frozenset((
                    jugador["id"],
                    rival_cercano["id"],
                ))

                if par in pares_mostrados:
                    continue

                pares_mostrados.add(par)

                distancias_jugadores.append((
                    distancia,
                    jugador,
                    rival_cercano,
                ))

            distancias_jugadores.sort(key=lambda item: item[0])

            # Dibujar las ocho parejas mas cercanas.
            for distancia, jugador, rival in distancias_jugadores[:8]:
                punto_a = (int(jugador["x"]), int(jugador["y"]))
                punto_b = (int(rival["x"]), int(rival["y"]))

                cv2.line(
                    frame,
                    punto_a,
                    punto_b,
                    YELLOW,
                    1,
                    cv2.LINE_AA,
                )

                centro_x = (punto_a[0] + punto_b[0]) // 2
                centro_y = (punto_a[1] + punto_b[1]) // 2

                texto_distancia = (
                    f"{jugador['id']} vs {rival['id']}: "
                    f"{distancia:.0f} px"
                )

                # Fondo oscuro para facilitar la lectura.
                (text_width, text_height), _ = cv2.getTextSize(
                    texto_distancia,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    1,
                )

                cv2.rectangle(
                    frame,
                    (centro_x, centro_y - text_height - 5),
                    (centro_x + text_width + 4, centro_y + 4),
                    DARK,
                    -1,
                )

                cv2.putText(
                    frame,
                    texto_distancia,
                    (centro_x + 2, centro_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    YELLOW,
                    1,
                    cv2.LINE_AA,
                )

            # ------------------------------------------------
            # 8. DISTANCIA DE CADA JUGADOR A LA PELOTA
            # ------------------------------------------------
            if ball_position is not None:
                distancias_balon = []

                for jugador in posiciones_jugadores:
                    distancia_balon = float(
                        np.hypot(
                            jugador["x"] - ball_position[0],
                            jugador["y"] - ball_position[1],
                        )
                    )

                    distancias_balon.append(
                        (distancia_balon, jugador)
                    )

                distancias_balon.sort(key=lambda item: item[0])

                # Mostrar las tres personas mas cercanas a la pelota.
                for distancia, jugador in distancias_balon[:3]:
                    texto = (
                        f"J{jugador['id']}-pelota: "
                        f"{distancia:.0f} px"
                    )

                    cv2.putText(
                        frame,
                        texto,
                        (max(10, width - 260),
                         35 + 24 * distancias_balon.index(
                             (distancia, jugador)
                         )),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        YELLOW,
                        1,
                        cv2.LINE_AA,
                    )

            # ------------------------------------------------
            # 9. MOSTRAR ANALISIS DE LA JUGADA
            # ------------------------------------------------
            if analisis_jugada is not None:
                titulo, detalle, color = analisis_jugada

                etiqueta_y = min(220, max(10, height - 100))
                etiqueta_ancho = min(width - 20, 720)
                etiqueta_alto = 78

                cv2.rectangle(
                    frame,
                    (10, etiqueta_y),
                    (10 + etiqueta_ancho, etiqueta_y + etiqueta_alto),
                    DARK,
                    -1,
                )

                cv2.putText(
                    frame,
                    titulo,
                    (22, etiqueta_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    detalle[:85],
                    (22, etiqueta_y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    WHITE,
                    1,
                    cv2.LINE_AA,
                )

            # ------------------------------------------------
            # 10. TIEMPO Y GUARDADO DEL FOTOGRAMA
            # ------------------------------------------------
            elapsed_seconds = frame_num / fps
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)

            time_text = (
                f"{minutes:02d}:{seconds:02d} | "
                f"Frame {frame_num + 1}/{total_frames}"
            )

            cv2.putText(
                frame,
                time_text,
                (max(10, width - 330), height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                WHITE,
                2,
                cv2.LINE_AA,
            )

            # Guardar al final, después de todas las anotaciones.
            writer.write(frame)

            if (frame_num + 1) % 100 == 0:
                print(
                    f"Video tactico: "
                    f"{frame_num + 1}/{total_frames} fotogramas"
                )

    finally:
        writer.release()

    print("Video tactico generado:", output_path)
    

# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def analizar_jugada_en_frame(players, ball_position, possession, width, height):
    """
    Genera una etiqueta táctica heurística para mostrar en el video.
    No confirma por sí sola errores tácticos reales.
    """
    if ball_position is None or possession not in (
        TEAM_ARGENTINA,
        TEAM_ENGLAND,
    ):
        return None

    equipo_atacante = possession
    equipo_defensor = (
        TEAM_ENGLAND
        if equipo_atacante == TEAM_ARGENTINA
        else TEAM_ARGENTINA
    )

    bx, by = ball_position

    atacantes = []
    defensores = []

    for track_id, player in players.items():
        if player.get("role") == "referee":
            continue

        team = player.get("team")
        position = get_player_position(player)

        if position is None:
            continue

        px, py = position
        distancia = float(np.hypot(px - bx, py - by))

        item = {
            "id": track_id,
            "x": float(px),
            "y": float(py),
            "distancia": distancia,
        }

        if team == equipo_atacante:
            atacantes.append(item)
        elif team == equipo_defensor:
            defensores.append(item)

    if not atacantes or not defensores:
        return None

    # Distancias aproximadas en píxeles; deben calibrarse por video.
    defensores_cerca = [
        p for p in defensores
        if p["distancia"] < width * 0.045
    ]

    # Un atacante cerca de la pelota y sin un defensor cercano
    # puede indicar una situación de peligro, no necesariamente un error.
    atacantes_libres = []

    for atacante in atacantes:
        distancia_balon = atacante["distancia"]

        if distancia_balon > width * 0.18:
            continue

        rivales_cerca = [
            defensor for defensor in defensores
            if np.hypot(
                defensor["x"] - atacante["x"],
                defensor["y"] - atacante["y"],
            ) < width * 0.055
        ]

        if not rivales_cerca:
            atacantes_libres.append(atacante)

    # Evita generar mensajes si no hay indicios relevantes.
    if len(defensores_cerca) >= 3:
        return (
            "ACIERTO DEFENSIVO POSIBLE",
            "Varios defensores presionan cerca del balon",
            GREEN,
        )

    if atacantes_libres:
        mas_cercano = min(
            atacantes_libres,
            key=lambda p: p["distancia"],
        )

        return (
            "PELIGRO OFENSIVO POSIBLE",
            f"Atacante {mas_cercano['id']} sin marca cercana",
            YELLOW,
        )

    if len(defensores_cerca) == 0:
        return (
            "PRESION DEFENSIVA BAJA",
            "No hay defensores cerca de la pelota",
            RED,
        )

    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STUBS_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"No se encontro el video: {VIDEO_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontro el modelo: {MODEL_PATH}")

    print(f"Clip: {CLIP_NAME}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Modelo: {MODEL_PATH}")

    # --------------------------------------------------------
    # 1. VIDEO
    # --------------------------------------------------------
    print("1/6 - Cargando video...")
    frames = read_video(str(VIDEO_PATH))

    if not frames:
        raise RuntimeError("No se pudieron leer los fotogramas.")

    print(f"Fotogramas: {len(frames)}")

    # --------------------------------------------------------
    # 2. TRACKING
    # --------------------------------------------------------
    print("2/6 - Detectando y siguiendo jugadores...")
    tracker = Tracker(str(MODEL_PATH))

    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=False,
        stub_path=str(TRACKS_STUB),
    )

    # Chequeo de consistencia
    if len(tracks["players"]) != len(frames):
        print(
            f"AVISO: tracks['players']={len(tracks['players'])} "
            f"!= frames={len(frames)}"
        )

    if len(frames) > 105 and len(tracks["players"]) > 105:
        save_tracking_debug(
            frames[105],
            tracks["players"][105],
            DEBUG_TRACKING_PATH,
        )
        print("Imagen de diagnostico guardada.")

    tracker.add_position_to_tracks(tracks)

    print("Frames del video:", len(frames))
    print("Frames con seguimiento:", len(tracks["players"]))

    for i in range(min(5, len(tracks["players"]))):
        print(f"Frame {i}: {len(tracks['players'][i])} jugadores detectados")

    # --------------------------------------------------------
    # 3. EQUIPOS
    # --------------------------------------------------------
    print("3/6 - Asignando equipos...")
    team_assigner = TeamAssigner()

    if tracks["players"] and tracks["players"][0]:
        team_assigner.assign_team_color(
            frames[0],
            tracks["players"][0],
        )

    if len(frames) != len(tracks["players"]):
        print(f"AVISO: frames={len(frames)}, seguimientos={len(tracks['players'])}")

    # Primera pasada: votos provisionales
    print("3a/6 - Votacion provisional...")
    team_votes_provisional = collect_team_votes(tracks, team_assigner, frames)
    resolved_provisional = resolve_track_teams(team_votes_provisional)
    apply_resolved_teams(tracks, resolved_provisional)

    goalkeeper_map_provisional = identify_goalkeepers(tracks)
    goalkeeper_ids_provisional = {
        tid for tid in goalkeeper_map_provisional.values() if tid is not None
    }
    print("Arqueros provisionales:", goalkeeper_map_provisional)

    # Segunda pasada: votacion final
    print("3b/6 - Votacion final...")
    team_assigner.player_team_dict.clear()
    team_assigner.player_team_history.clear()

    team_votes = collect_team_votes(tracks, team_assigner, frames)
    resolved_teams = resolve_track_teams(team_votes)
    apply_resolved_teams(tracks, resolved_teams)

    # --------------------------------------------------------
    # 4. POSICIONES Y ROLES
    # --------------------------------------------------------
    print("4/6 - Ajustando posiciones y roles...")

    camera_estimator = CameraMovementEstimator(frames[0])

    camera_movement = camera_estimator.get_camera_movement(
        frames,
        read_from_stub=False,     # <- forzar recalculo por seguridad
        stub_path=str(CAMERA_STUB),
    )

    print(f"camera_movement len = {len(camera_movement)}")
    print(f"frames len = {len(frames)}")

    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # 4a. Arqueros definitivos
    goalkeeper_map = identify_goalkeepers(tracks)
    goalkeeper_ids = {
        tid for tid in goalkeeper_map.values() if tid is not None
    }
    print("Arqueros detectados:", goalkeeper_map)

    # 4b. Consolidar roles
    for frame_players in tracks["players"]:
        for track_id, player in frame_players.items():
            team = player.get("team", TEAM_NEUTRAL)
            role = player.get("role", "field_player")

            if role == "referee":
                player["team"] = TEAM_NEUTRAL
                player["team_color"] = REFEREE_COLOR
                continue

            if role == "goalkeeper":
                if team in (TEAM_ARGENTINA, TEAM_ENGLAND):
                    player["team_color"] = GOALKEEPER_COLOR
                else:
                    player["team"] = TEAM_NEUTRAL
                    player["role"] = "neutral"
                    player["team_color"] = TEAM_COLORS[TEAM_NEUTRAL]
                continue

            if team not in (TEAM_ARGENTINA, TEAM_ENGLAND):
                player["team"] = TEAM_NEUTRAL
                player["role"] = "neutral"
                player["team_color"] = TEAM_COLORS[TEAM_NEUTRAL]

    # Contar arbitros detectados (diagnostico)
    referee_tracks = set()
    for frame_players in tracks["players"]:
        for tid, p in frame_players.items():
            if p.get("role") == "referee":
                referee_tracks.add(tid)
    print(f"Tracks de arbitros detectados: {referee_tracks}")

    # --------------------------------------------------------
    # 5. METRICAS Y VIDEO
    # --------------------------------------------------------
    print("5/6 - Calculando metricas y generando video...")


    team_ball_control = estimate_possession(tracks)

    # ANALISIS DEFENSIVO
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not fps or fps <= 0:
        fps = 25.0

    defensive_alerts = detectar_alertas_defensivas(
        tracks=tracks,
        fps=fps,
        salida_csv=OUTPUT_DIR / f"defensa_{CLIP_NAME}.csv",
        tiempos_goles=GOAL_TIMES_SECONDS,
    )

    # CONTINUA EL INFORME TACTICO
    report = build_tactical_report(tracks, team_ball_control)

    print(f"team_ball_control len = {len(team_ball_control)}")

    report = build_tactical_report(tracks, team_ball_control)

    report.to_csv(REPORT_PATH, index=False)
    create_summary(report).to_csv(SUMMARY_PATH, index=False)
    create_tactical_map(tracks, MAP_PATH)

    create_tactical_video(
        frames,
        tracks,
        team_ball_control,
        TACTICAL_VIDEO_PATH,
    )

    # --------------------------------------------------------
    # 6. RESULTADOS
    # --------------------------------------------------------
    print("6/6 - Resultados")
    print(f"Video tactico:      {TACTICAL_VIDEO_PATH}")
    print(f"Informe detallado:  {REPORT_PATH}")
    print(f"Resumen tactico:    {SUMMARY_PATH}")
    print(f"Mapa tactico:       {MAP_PATH}")
    print()
    print("Primeras filas del informe:")
    print(report.head(10).to_string(index=False))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    main()