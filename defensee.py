"""Analisis temporal heuristico de pases y situaciones defensivas.



Las distancias se calculan en pixeles de imagen. No se presentan como metros

sin una homografia calibrada para el video. Los eventos son candidatos para

revision/etiquetado, no veredictos tacticos automaticos.

"""

from __future__ import annotations



import csv

import math

from pathlib import Path

from typing import Any, Iterable

import numpy as np

TEAM_IDS = (1, 2)

REFEREE_ROLES = {"referee", "arbitro", "official"}

GOALKEEPER_ROLES = {"goalkeeper", "goalie", "arquero"}





def _finite(value: Any) -> bool:

    try:

        return math.isfinite(float(value))

    except (TypeError, ValueError):

        return False





def _bbox(obj: Any):

    if obj is None:

        return None

    if isinstance(obj, dict):

        for key in ("bbox", "xyxy", "box"):

            value = obj.get(key)

            try:

                vals = list(value)

                if len(vals) >= 4 and all(_finite(v) for v in vals[:4]):

                    x1, y1, x2, y2 = map(float, vals[:4])

                    if x2 > x1 and y2 > y1:

                        return [x1, y1, x2, y2]

            except (TypeError, ValueError):

                pass

        for key in ("detection", "data"):

            if key in obj:

                result = _bbox(obj[key])

                if result is not None:

                    return result

    for attr in ("xyxy", "bbox", "box"):

        if hasattr(obj, attr):

            value = getattr(obj, attr)

            try:

                if hasattr(value, "tolist"):

                    value = value.tolist()

                while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):

                    value = value[0]

                vals = list(value)

                if len(vals) >= 4 and all(_finite(v) for v in vals[:4]):

                    x1, y1, x2, y2 = map(float, vals[:4])

                    if x2 > x1 and y2 > y1:

                        return [x1, y1, x2, y2]

            except (TypeError, ValueError):

                pass

    if isinstance(obj, (list, tuple)) and len(obj) >= 4 and all(_finite(v) for v in obj[:4]):

        x1, y1, x2, y2 = map(float, obj[:4])

        if x2 > x1 and y2 > y1:

            return [x1, y1, x2, y2]

    return None





def _position(obj: Any, is_ball: bool = False):

    # Se usan coordenadas de imagen para todos: jugador por los pies, balon por centro.

    if isinstance(obj, dict):

        box = _bbox(obj)

        if box is not None:

            x1, y1, x2, y2 = box

            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0) if is_ball else ((x1 + x2) / 2.0, y2)

        for key in ("position", "position_adjusted"):

            val = obj.get(key)

            if val is not None:

                try:

                    x, y = float(val[0]), float(val[1])

                    if math.isfinite(x) and math.isfinite(y):

                        return (x, y)

                except (TypeError, ValueError, IndexError):

                    pass

    box = _bbox(obj)

    if box is not None:

        x1, y1, x2, y2 = box

        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0) if is_ball else ((x1 + x2) / 2.0, y2)

    return None





def _team(obj: Any):

    if not isinstance(obj, dict):

        return None

    value = obj.get("team", obj.get("team_id", obj.get("team_number")))

    try:

        value = int(value)

    except (TypeError, ValueError):

        return None

    return value if value in TEAM_IDS else None





def _role(obj: Any) -> str:

    return str(obj.get("role", "field_player")).strip().lower() if isinstance(obj, dict) else "field_player"





def _items(frame_data: Any):

    if isinstance(frame_data, dict):

        return list(frame_data.items())

    if isinstance(frame_data, (list, tuple)):

        return list(enumerate(frame_data))

    return []





def _ball_for_frame(frame_data: Any):

    for _, obj in _items(frame_data):

        pos = _position(obj, is_ball=True)

        if pos is not None:

            return pos

    return _position(frame_data, is_ball=True)





def _players_for_frame(frame_data: Any):

    result = []

    for track_id, obj in _items(frame_data):

        role = _role(obj)

        team = _team(obj)

        if role in REFEREE_ROLES or team is None:

            continue

        pos = _position(obj)

        box = _bbox(obj)

        if pos is None or box is None:

            continue

        result.append({

            "id": track_id,

            "team": team,

            "role": role,

            "pos": (float(pos[0]), float(pos[1])),

            "height": max(1.0, box[3] - box[1]),

        })

    return result





def _distance(a, b) -> float:

    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

def _world_point(view_transformer, point, frame_num=None):
    """Convierte un punto a metros; compatible con homografías estáticas y por frame."""
    if view_transformer is None or not getattr(view_transformer, "enabled", False) or point is None:
        return None
    try:
        try:
            result = view_transformer.transform_point(point, frame_num=frame_num) if frame_num is not None else view_transformer.transform_point(point)
        except TypeError:
            result = view_transformer.transform_point(point)
        if result is None:
            return None
        result = np.asarray(result, dtype=float).reshape(-1)
        if result.size < 2 or not np.isfinite(result[:2]).all():
            return None
        return result[:2]
    except (TypeError, ValueError, IndexError, AttributeError):
        return None






def _point_segment_distance(point, start, end):

    px, py = point

    sx, sy = start

    ex, ey = end

    dx, dy = ex - sx, ey - sy

    denom = dx * dx + dy * dy

    if denom <= 1e-9:

        return _distance(point, start), 0.0

    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))

    proj = (sx + t * dx, sy + t * dy)

    return _distance(point, proj), t





def _nearest_player(players, ball_pos):

    if not players or ball_pos is None:

        return None, float("inf")

    player = min(players, key=lambda p: _distance(p["pos"], ball_pos))

    return player, _distance(player["pos"], ball_pos)





def _infer_possession(players, ball_pos, last_team, scale):

    nearest, distance = _nearest_player(players, ball_pos)

    if nearest is not None and distance <= 3.2 * scale:

        return nearest["team"]

    return last_team if last_team in TEAM_IDS else None





# NUEVO: helper para calcular velocidad de pase en px/s y, si hay homografia,

# en m/s. Devuelve (vel_px_s, vel_m_s_o_None).



def _pass_speed(

    ball_start,

    ball_end,

    dt,

    view_transformer=None,

    start_frame=None,

    end_frame=None,

):

    """Calcula velocidad en px/s y, si es posible, en m/s."""

    travel_px = _distance(ball_start, ball_end)

    vel_px_s = travel_px / dt if dt > 1e-9 else None

    vel_m_s = None



    if (

        view_transformer is None

        or not getattr(view_transformer, "enabled", False)

        or dt <= 1e-9

    ):

        return vel_px_s, vel_m_s



    w_start = _world_point(view_transformer, ball_start, start_frame)
    w_end = _world_point(view_transformer, ball_end, end_frame)
    if w_start is None or w_end is None:
        return vel_px_s, None
    dist_m = float(np.linalg.norm(w_end - w_start))
    vel_m_s = dist_m / dt
    if not math.isfinite(vel_m_s):
        vel_m_s = None

    return vel_px_s, vel_m_s



def detectar_alertas_defensivas(

    tracks: dict,

    fps: float,

    salida_csv: str | Path,

    tiempos_goles: Iterable[float] | None = None,

    equipo_en_posesion: Any = None,

    view_transformer: Any = None,   # NUEVO: opcional, para pasar a m/s

) -> list[dict]:

    """Detecta candidatos temporales de pase/intercepcion y alertas defensivas.



    Mantiene la firma usada por futbol.py. Los umbrales se normalizan por la

    altura mediana detectada del jugador y los resultados se guardan en CSV.

    """

    if not isinstance(tracks, dict):

        raise TypeError("tracks debe contener 'players' y 'ball'.")

    try:

        fps = float(fps)

    except (TypeError, ValueError):

        fps = 25.0

    if not math.isfinite(fps) or fps <= 0:

        fps = 25.0



    players_frames = tracks.get("players", [])

    ball_frames = tracks.get("ball", [])

    n_frames = min(len(players_frames), len(ball_frames))

    if not n_frames:

        _write_csv(Path(salida_csv), [])

        return []



    frame_players, frame_balls, frame_scales, possessions = [], [], [], []

    last_team = None

    for i in range(n_frames):

        players = _players_for_frame(players_frames[i])

        ball = _ball_for_frame(ball_frames[i])

        heights = [p["height"] for p in players if p["role"] not in GOALKEEPER_ROLES]

        scale = max(12.0, float(sorted(heights)[len(heights)//2])) if heights else 24.0

        frame_players.append(players)

        frame_balls.append(ball)

        frame_scales.append(scale)

        provided = None

        try:

            v = equipo_en_posesion[i]

            v = int(v)

            if v in TEAM_IDS:

                provided = v

        except (TypeError, ValueError, IndexError, KeyError):

            pass

        poss = provided if provided is not None else _infer_possession(players, ball, last_team, scale)

        possessions.append(poss)

        if poss in TEAM_IDS:

            last_team = poss



    events = []

    last_event_frame = -10**9

    window = max(2, int(round(0.22 * fps)))

    cooldown = max(1, int(round(0.65 * fps)))



    for i in range(window, n_frames):

        ball_start, ball_end = frame_balls[i-window], frame_balls[i]

        if ball_start is None or ball_end is None:

            continue

        start_team = possessions[i-window]

        if start_team not in TEAM_IDS or i - last_event_frame < cooldown:

            continue

        scale = max(frame_scales[i-window], frame_scales[i])

        travel = _distance(ball_start, ball_end)

        # Filtra vibración de detección y movimiento mínimo del balón.

        if travel < max(10.0, 0.85 * scale):

            continue



        start_players, end_players = frame_players[i-window], frame_players[i]

        start_near, start_dist = _nearest_player(start_players, ball_start)

        end_near, end_dist = _nearest_player(end_players, ball_end)

        if start_near is None or end_near is None or start_near["team"] != start_team:

            continue

        if start_dist > 3.2 * scale or end_dist > 3.0 * scale:

            continue



        opposing = [p for p in end_players if p["team"] != start_team]

        # El arquero cuenta para detectar una posible recuperacion, pero no se

        # confunde con un defensor de campo al evaluar la linea defensiva.

        opposing_field = [p for p in opposing if p["role"] not in GOALKEEPER_ROLES]

        end_team = end_near["team"]

        event_type = None

        tags = []

        detail = f"Balon recorrio {travel:.0f}px en {window/fps:.2f}s"

        nearest_defender = min((_distance(p["pos"], end_near["pos"]) for p in opposing_field), default=float("inf"))
        pass_distance_m = None
        defender_receiver_m = None
        wbs = _world_point(view_transformer, ball_start, i - window)
        wbe = _world_point(view_transformer, ball_end, i)
        if wbs is not None and wbe is not None:
            pass_distance_m = float(np.linalg.norm(wbe - wbs))
        closest_defender = min(
            opposing_field,
            key=lambda p: _distance(p["pos"], end_near["pos"]),
            default=None,
        )
        if closest_defender is not None:
            wd = _world_point(view_transformer, closest_defender["pos"], i)
            wr = _world_point(view_transformer, end_near["pos"], i)
            if wd is not None and wr is not None:
                defender_receiver_m = float(np.linalg.norm(wd - wr))


        defenders_crossed = 0

        for defender in opposing_field:

            seg_dist, t = _point_segment_distance(defender["pos"], ball_start, ball_end)

            if 0.12 < t < 0.92 and seg_dist < 0.9 * scale:

                defenders_crossed += 1



        # NUEVO: velocidad del pase (px/s y, si hay homografia, m/s)

        dt_pass = max(1e-6, window / fps)



        vel_px_s, vel_m_s = _pass_speed(

            ball_start,

            ball_end,

            dt_pass,

            view_transformer=view_transformer,

            start_frame=i - window,

            end_frame=i,

        )



        if end_team != start_team:

            event_type = "intercepcion_defensiva_posible"

            tags.append("equipo_rival_recupera_cerca_del_final_del_pase")

            detail += f"; posible recuperacion rival J{end_near['id']}"

        elif end_near["id"] != start_near["id"]:

            completed = True

            if defenders_crossed >= 1 and nearest_defender > 1.2 * scale:

                event_type = "pase_filtrado_posible"

                tags.extend(["pase_completado", f"defensores_cortados={defenders_crossed}", "acierto_ofensivo_posible"])

            elif nearest_defender > 2.5 * scale:

                event_type = "error_defensivo_posible"

                tags.extend(["pase_completado", "receptor_con_espacio", "acierto_ofensivo_posible"])

            else:

                event_type = "pase_completado_posible"

                tags.append("recepcion_de_companero_detectada")

            detail += f"; J{start_near['id']} -> J{end_near['id']}; marca mas cercana {nearest_defender:.0f}px"

        else:

            # Balón desplazado sin cambio claro de receptor: no se inventa un pase.

            continue



        # NUEVO: enriquecer el detalle con la velocidad del pase

        if vel_m_s is not None:

            detail += f"; vel {vel_m_s:.1f} m/s ({vel_m_s*3.6:.1f} km/h)"

        elif vel_px_s is not None:

            detail += f"; vel {vel_px_s:.0f} px/s"



        events.append({

            "inicio_seconds": round(max(0.0, (i-window)/fps), 3),

            "fin_seconds": round(i/fps, 3),

            "frame_inicio": i-window,

            "frame_pico": i,

            "equipo_atacante": start_team,

            "tipo_evento": event_type,

            "score": min(95, 55 + 10*min(defenders_crossed, 3) + (10 if nearest_defender > 2.5*scale else 0)),

            "jugador_pasador": start_near["id"],

            "jugador_receptor_o_recuperador": end_near["id"],

            "distancia_pase_px": round(travel, 2),
            "distancia_pase_m": round(pass_distance_m, 2) if pass_distance_m is not None else "",

            "distancia_defensor_receptor_px": round(nearest_defender, 2) if math.isfinite(nearest_defender) else "",
            "distancia_defensor_receptor_m": round(defender_receiver_m, 2) if defender_receiver_m is not None else "",

            "defensores_cruzados_estimados": defenders_crossed,

            "razones": ";".join(tags),

            "detalle": detail,

            # NUEVO: columnas de velocidad de pase

            "velocidad_pase_px_s": round(vel_px_s, 2) if vel_px_s is not None else "",

            "velocidad_pase_m_s": round(vel_m_s, 2) if vel_m_s is not None else "",
            "velocidad_pase_km_h": round(vel_m_s * 3.6, 2) if vel_m_s is not None else "",

            "gol_cercano_seconds": "",

            "relacion_con_gol": "",

        })

        last_event_frame = i



    # Alertas estáticas: receptor libre cerca de la zona del balón, sin adjudicar culpa.

    for i in range(n_frames):

        if possessions[i] not in TEAM_IDS or frame_balls[i] is None:

            continue

        players = frame_players[i]

        attackers = [p for p in players if p["team"] == possessions[i] and p["role"] not in GOALKEEPER_ROLES]

        defenders = [p for p in players if p["team"] != possessions[i] and p["role"] not in GOALKEEPER_ROLES]

        if not attackers or not defenders:

            continue

        scale = frame_scales[i]

        free = []

        for a in attackers:

            if _distance(a["pos"], frame_balls[i]) > 9.0 * scale:

                continue

            mark = min(_distance(d["pos"], a["pos"]) for d in defenders)

            if mark > 2.5 * scale:

                free.append((mark, a))

        if not free:

            continue

        mark, receiver = max(free, key=lambda item: item[0])

        t = i / fps

        # Evita llenar el CSV con una alerta por cada frame de la misma situación.

        if any(e.get("tipo_evento") == "receptor_libre_posible" and

               abs(float(e["inicio_seconds"]) - t) < 0.8 for e in events):

            continue

        if any(e.get("tipo_evento") == "receptor_libre_posible" and

               float(e["inicio_seconds"]) <= t <= float(e["fin_seconds"]) + 0.8 for e in events):

            continue

        events.append({

            "inicio_seconds": round(t, 3), "fin_seconds": round(t, 3),

            "frame_inicio": i, "frame_pico": i,

            "equipo_atacante": possessions[i],

            "tipo_evento": "receptor_libre_posible", "score": 45,

            "jugador_pasador": "", "jugador_receptor_o_recuperador": receiver["id"],

            "distancia_pase_px": "", "distancia_pase_m": "",
            "distancia_defensor_receptor_px": round(mark, 2), "distancia_defensor_receptor_m": "",

            "defensores_cruzados_estimados": 0,

            "razones": "receptor_sin_marca_cercana;revisar_linea_de_pase",

            "detalle": f"J{receiver['id']} libre; defensor mas cercano a {mark:.0f}px",

            "velocidad_pase_px_s": "",

            "velocidad_pase_m_s": "", "velocidad_pase_km_h": "",

            "gol_cercano_seconds": "", "relacion_con_gol": "",

        })



    goals = sorted(float(t) for t in (tiempos_goles or []) if _finite(t) and float(t) >= 0)

    for event in events:

        start = float(event["inicio_seconds"])

        next_goal = next((g for g in goals if start <= g <= start + 10.0), None)

        if next_goal is not None:

            event["gol_cercano_seconds"] = next_goal

            event["relacion_con_gol"] = "revisar_evento_antes_del_gol"

    events.sort(key=lambda e: float(e["inicio_seconds"]))

    _write_csv(Path(salida_csv), events)

    print(f"[defensee.py] Eventos candidatos guardados: {len(events)} -> {salida_csv}")

    return events





def _write_csv(path: Path, rows: list[dict]):

    path.parent.mkdir(parents=True, exist_ok=True)

    # NUEVO: se agregaron las columnas de velocidad de pase

    columns = [

        "inicio_seconds", "fin_seconds", "frame_inicio", "frame_pico",

        "equipo_atacante", "tipo_evento", "score", "jugador_pasador",

        "jugador_receptor_o_recuperador", "distancia_pase_px", "distancia_pase_m",

        "distancia_defensor_receptor_px", "distancia_defensor_receptor_m", "defensores_cruzados_estimados",

        "velocidad_pase_px_s", "velocidad_pase_m_s", "velocidad_pase_km_h",

        "razones", "detalle", "gol_cercano_seconds", "relacion_con_gol",

    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")

        writer.writeheader()

        writer.writerows(rows)





# Alias compatible con integraciones anteriores.

detect_defensive_alerts = detectar_alertas_defensivas