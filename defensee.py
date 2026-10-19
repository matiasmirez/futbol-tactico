"""Reconstrucción heurística de secuencias de ataque a partir del tracking.

Detecta candidatos a conducción, pase/recepción, remate, pérdida y posible gol.
No usa minutos introducidos manualmente ni escribe CSV. Los resultados son
estimaciones: con tracking de cajas 2D no siempre puede confirmarse posesión,
pase, remate o gol de forma inequívoca.
"""
from __future__ import annotations

import math
from collections import defaultdict
import numpy as np

TEAM_ARGENTINA = 1
TEAM_ENGLAND = 2
TEAM_NEUTRAL = 0
ALTURA_JUGADOR_M = 1.75
MAX_SPEED_M_S = 12.0  # Tope de velocidad humana (~43 km/h)


def _valid_point(value):
    if value is None:
        return None
    try:
        p = np.asarray(value, dtype=float).reshape(-1)[:2]
        if len(p) == 2 and np.isfinite(p).all():
            return p
    except (TypeError, ValueError):
        pass
    return None


def _bbox(player):
    box = player.get("bbox")
    if box is None or len(box) < 4:
        return None
    try:
        b = np.asarray(box[:4], dtype=float)
        if not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
            return None
        return b
    except (TypeError, ValueError):
        return None


def _player_point(player):
    b = _bbox(player)
    if b is not None:
        return np.array([(b[0] + b[2]) / 2.0, b[3]], dtype=float)
    transformed = _valid_point(player.get("position_transformed"))
    if transformed is not None:
        return transformed
    return _valid_point(player.get("position"))


def _ball_point(ball):
    b = _bbox(ball)
    if b is not None:
        return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=float)
    return _valid_point(ball.get("position_transformed")) if _valid_point(ball.get("position_transformed")) is not None else _valid_point(ball.get("position"))


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _transform(transformer, point, frame_num):
    if transformer is None or point is None or not getattr(transformer, "enabled", False):
        return None
    try:
        try:
            out = transformer.transform_point(point, frame_num=frame_num)
        except TypeError:
            out = transformer.transform_point(point)
        return _valid_point(out)
    except Exception:
        return None


def _metric_point(player, raw_point, transformer, frame_num):
    transformed = _transform(transformer, raw_point, frame_num)
    if transformed is not None:
        return transformed, "m"
    box = _bbox(player)
    if box is None:
        return None, "px"
    height = float(box[3] - box[1])
    if height <= 1:
        return None, "px"
    return np.asarray(raw_point, dtype=float) * (ALTURA_JUGADOR_M / height), "estimado_m"


def _owner(players, ball_point, max_factor=1.65):
    if ball_point is None:
        return None
    choices = []
    for tid, player in players.items():
        if player.get("role", "field_player") not in ("field_player", "goalkeeper"):
            continue
        team = player.get("team", TEAM_NEUTRAL)
        if team not in (TEAM_ARGENTINA, TEAM_ENGLAND):
            continue
        point = _player_point(player)
        box = _bbox(player)
        if point is None or box is None:
            continue
        d = _distance(point, ball_point)
        h = max(1.0, float(box[3] - box[1]))
        choices.append((d / h, d, tid, player))
    if not choices:
        return None
    ratio, _, tid, player = min(choices, key=lambda x: x[0])
    return (tid, player) if ratio <= max_factor else None


def _unificar_trayectorias_y_equipos(tracks, id_original, frame_remate, ventana_previa_s, fps):
    """
    Costura de identidad (Identity Stitching) con escala dinámica.
    """
    players_by_frame = tracks.get("players", [])
    equipo_original = None
    pos_remate = None
    altura_px_remate = 50.0
    
    if frame_remate < len(players_by_frame):
        p_remate = players_by_frame[frame_remate].get(id_original)
        if p_remate:
            equipo_original = p_remate.get("team")
            pos_remate = _player_point(p_remate)
            b = _bbox(p_remate)
            if b is not None:
                altura_px_remate = max(1.0, float(b[3] - b[1]))
            
    if equipo_original is None or pos_remate is None:
        return [id_original]

    ids_conocidos = [id_original]
    limite = max(0, frame_remate - int(ventana_previa_s * fps))
    m_por_px = ALTURA_JUGADOR_M / altura_px_remate
    
    for f in range(frame_remate - 1, limite - 1, -1):
        frame_players = players_by_frame[f]
        candidato_id = None
        min_dist_metros = float('inf')
        
        for tid, p in frame_players.items():
            if tid in ids_conocidos: 
                continue
            if p.get("team") != equipo_original:
                continue
                
            pt = _player_point(p)
            if pt is None: 
                continue
            
            dist_px = _distance(pt, pos_remate)
            dist_metros = dist_px * m_por_px
            
            frames_gap = frame_remate - f
            max_metros_permitidos = frames_gap * 0.5 # 0.5m por frame
            
            if dist_metros <= max_metros_permitidos and dist_metros < min_dist_metros:
                min_dist_metros = dist_metros
                candidato_id = tid
                
        if candidato_id is not None:
            ids_conocidos.append(candidato_id)
            pos_remate = _player_point(players_by_frame[f][candidato_id])
            b_cand = _bbox(players_by_frame[f][candidato_id])
            if b_cand is not None:
                altura_px_remate = max(1.0, float(b_cand[3] - b_cand[1]))
                m_por_px = ALTURA_JUGADOR_M / altura_px_remate
            
    return ids_conocidos


def detectar_alertas_defensivas(
    tracks, fps=25.0, salida_csv=None, tiempos_goles=None,
    equipo_en_posesion=None, view_transformer=None, bbox_arco=None,
    **kwargs,
):
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    players_by_frame = tracks.get("players", [])
    balls_by_frame = tracks.get("ball", [])
    total = min(len(players_by_frame), len(balls_by_frame))
    if total < 2:
        return []

    rows = []
    prev_ball = None
    prev_ball_frame = None
    for i in range(total):
        players = players_by_frame[i] or {}
        ball = (balls_by_frame[i] or {}).get(1, {})
        ball_px = _ball_point(ball)
        owner_info = _owner(players, ball_px)
        owner_id, owner = owner_info if owner_info else (None, None)
        team = owner.get("team", TEAM_NEUTRAL) if owner else TEAM_NEUTRAL
        ball_metric = _transform(view_transformer, ball_px, i)
        rows.append({
            "frame": i, "time": i / fps, "ball_px": ball_px,
            "ball_metric": ball_metric, "owner_id": owner_id,
            "owner": owner, "team": team,
        })

    events = []
    carry_start = None
    carry_owner = None
    carry_team = None
    carry_start_point = None
    carry_last_point = None
    carry_last_frame = None
    carry_distance = 0.0
    team_carry_total = defaultdict(float)
    attack_start = None
    attack_team = None
    attack_players = []
    pass_count = 0
    last_owner = None
    last_team = TEAM_NEUTRAL
    last_ball = None
    last_ball_frame = None
    last_event_frame = -999
    lost_frames = 0

    def emit(kind, start_frame, end_frame, detail, **fields):
        start_frame = max(0, int(start_frame))
        end_frame = max(start_frame, int(end_frame))
        row = {
            "tipo_evento": kind,
            "inicio_seconds": round(start_frame / fps, 3),
            "fin_seconds": round(end_frame / fps, 3),
            "inicio_frame": start_frame,
            "fin_frame": end_frame,
            "detalle": detail,
        }
        row.update(fields)
        events.append(row)

    for i, row in enumerate(rows):
        owner_id, owner, team = row["owner_id"], row["owner"], row["team"]
        ball_px = row["ball_px"]
        ball_speed_px_s = None
        ball_distance_px = None
        if ball_px is not None and last_ball is not None and last_ball_frame is not None:
            dt = (i - last_ball_frame) / fps
            if 0 < dt <= 0.8:
                ball_distance_px = _distance(ball_px, last_ball)
                ball_speed_px_s = ball_distance_px / dt
        if ball_px is not None:
            last_ball, last_ball_frame = ball_px.copy(), i

        if owner_id is not None and team in (TEAM_ARGENTINA, TEAM_ENGLAND):
            lost_frames = 0
            if attack_start is None or (attack_team is not None and team != attack_team):
                if attack_start is not None:
                    emit("perdida", max(attack_start, i - 8), i,
                         f"El equipo {attack_team} pierde o cede la posesión; final de secuencia no confirmado.",
                         equipo=attack_team, jugadores=list(dict.fromkeys(attack_players)),
                         cantidad_pases=pass_count, metros_conduccion_equipo=round(team_carry_total[attack_team], 2))
                attack_start, attack_team = i, team
                attack_players, pass_count = [], 0
                team_carry_total[team] = 0.0
            if owner_id not in attack_players:
                attack_players.append(owner_id)

            player_point = _player_point(owner)
            if carry_owner != owner_id:
                if carry_owner is not None and carry_distance > 0.5:
                    emit("conduccion", carry_start if carry_start is not None else i,
                         carry_last_frame if carry_last_frame is not None else i,
                         f"J{carry_owner} condujo la pelota; distancia estimada/calibrada.",
                         jugador_id=carry_owner, equipo=carry_team,
                         metros_conduccion=round(carry_distance, 2),
                         metros_conduccion_equipo=round(team_carry_total.get(carry_team, 0.0), 2))
                if last_owner is not None and owner_id != last_owner and last_team == team:
                    pass_count += 1
                    dist_m = None
                    if rows[i]["ball_metric"] is not None and i > 0 and rows[i-1]["ball_metric"] is not None:
                        dist_m = _distance(rows[i]["ball_metric"], rows[i-1]["ball_metric"])
                    detail = f"Posible pase de J{last_owner} a J{owner_id}; inferido por cambio de poseedor."
                    emit("pase_completado_posible", max(0, i - max(1, int(0.5 * fps))), i,
                         detail, pasador_id=last_owner, receptor_id=owner_id, equipo=team,
                         distancia_pase_m=round(dist_m, 2) if dist_m is not None else "",
                         distancia_pase_px=round(ball_distance_px, 1) if ball_distance_px is not None else "")
                    emit("recepcion", i, i, f"J{owner_id} parece recibir la pelota de J{last_owner}.",
                         jugador_id=owner_id, pasador_id=last_owner, equipo=team)
                carry_start, carry_owner, carry_team = i, owner_id, team
                carry_start_point = player_point.copy() if player_point is not None else None
                carry_last_point = player_point.copy() if player_point is not None else None
                carry_last_frame = i
                carry_distance = 0.0
            else:
                if player_point is not None and carry_last_point is not None and carry_last_frame is not None:
                    frame_gap = i - carry_last_frame
                    if 0 < frame_gap <= max(1, int(0.4 * fps)):
                        transformed_now = _transform(view_transformer, player_point, i)
                        transformed_prev = _transform(view_transformer, carry_last_point, carry_last_frame)
                        if transformed_now is not None and transformed_prev is not None:
                            step = _distance(transformed_now, transformed_prev)
                        else:
                            box = _bbox(owner)
                            h = float(box[3] - box[1]) if box is not None else 0.0
                            step = _distance(player_point, carry_last_point) * (ALTURA_JUGADOR_M / h) if h > 1 else 0.0
                        
                        # Tope de velocidad física para evitar saltos por zoom/ID switch
                        max_step = (frame_gap / fps) * MAX_SPEED_M_S
                        if np.isfinite(step) and 0 <= step <= max_step:
                            carry_distance += step
                            team_carry_total[team] += step
                    carry_last_point, carry_last_frame = player_point.copy(), i

            if ball_speed_px_s is not None and ball_speed_px_s > 650 and owner is not None:
                if i - last_event_frame > int(0.8 * fps):
                    speed_kmh = None
                    if ball_distance_px is not None and last_ball_frame is not None and i > 0:
                        prev_metric = rows[i - 1].get("ball_metric")
                        curr_metric = row.get("ball_metric")
                        if prev_metric is not None and curr_metric is not None:
                            dt_metric = 1.0 / fps
                            speed_kmh = _distance(curr_metric, prev_metric) / dt_metric * 3.6
                    emit("tiro_al_arco", max(0, i - 2), min(total - 1, i + int(0.4 * fps)),
                         f"Posible remate de J{owner_id}; velocidad aparente del balón {ball_speed_px_s:.0f} px/s. No confirma que vaya al arco.",
                         tirador_id=owner_id, equipo=team,
                         velocidad_tiro_px_s=round(ball_speed_px_s, 1),
                         velocidad_tiro_km_h=round(speed_kmh, 1) if speed_kmh is not None else None,
                         velocidad_tiro_metodo="calibración de cancha" if speed_kmh is not None else "no disponible sin calibración fiable")
                    last_event_frame = i
        else:
            lost_frames += 1
            if ball_speed_px_s is not None and ball_speed_px_s > 800 and i - last_event_frame > int(0.8 * fps):
                team_guess = last_team if last_team in (TEAM_ARGENTINA, TEAM_ENGLAND) else TEAM_NEUTRAL
                speed_kmh = None
                if i > 0 and rows[i - 1].get("ball_metric") is not None and row.get("ball_metric") is not None:
                    speed_kmh = _distance(row["ball_metric"], rows[i - 1]["ball_metric"]) * fps * 3.6
                emit("tiro_al_arco", max(0, i - 2), min(total - 1, i + int(0.5 * fps)),
                     f"Trayectoria rápida del balón sin poseedor claro ({ball_speed_px_s:.0f} px/s): posible remate o pase largo.",
                     tirador_id=last_owner if last_owner is not None else "", equipo=team_guess,
                     velocidad_tiro_px_s=round(ball_speed_px_s, 1),
                     velocidad_tiro_km_h=round(speed_kmh, 1) if speed_kmh is not None else None,
                     velocidad_tiro_metodo="calibración de cancha" if speed_kmh is not None else "no disponible sin calibración fiable")
                last_event_frame = i
            if carry_owner is not None and carry_distance > 0.5 and lost_frames >= max(2, int(0.3 * fps)):
                emit("conduccion", carry_start if carry_start is not None else i,
                     carry_last_frame if carry_last_frame is not None else i,
                     f"J{carry_owner} condujo antes de perderse el contacto con el balón.",
                     jugador_id=carry_owner, equipo=carry_team,
                     metros_conduccion=round(carry_distance, 2),
                     metros_conduccion_equipo=round(team_carry_total.get(carry_team, 0.0), 2))
                carry_owner, carry_team, carry_start = None, None, None
                carry_distance, carry_last_point, carry_last_frame = 0.0, None, None

        last_owner, last_team = owner_id, team if team in (TEAM_ARGENTINA, TEAM_ENGLAND) else last_team

    if carry_owner is not None and carry_distance > 0.5:
        emit("conduccion", carry_start if carry_start is not None else total - 1,
             carry_last_frame if carry_last_frame is not None else total - 1,
             f"J{carry_owner} condujo la pelota; fin del clip.", jugador_id=carry_owner,
             equipo=carry_team, metros_conduccion=round(carry_distance, 2),
             metros_conduccion_equipo=round(team_carry_total.get(carry_team, 0.0), 2))
    if attack_start is not None:
        emit("fin_ataque", attack_start, total - 1,
             f"Secuencia automática del equipo {attack_team}: {len(set(attack_players))} jugadores detectados, {pass_count} cambios de poseedor interpretados como pases. El desenlace no se confirma automáticamente.",
             equipo=attack_team, jugadores=list(dict.fromkeys(attack_players)),
             cantidad_pases=pass_count, metros_conduccion_equipo=round(team_carry_total[attack_team], 2))

    for j, row in enumerate(rows):
        if row["ball_px"] is None:
            continue
        if j + 1 >= total or rows[j + 1]["ball_px"] is not None:
            continue
        if j + 2 < total and rows[j + 2]["ball_px"] is not None:
            continue
        prev = rows[max(0, j - 3)]["ball_px"]
        if prev is None:
            continue
        pass

    events.sort(key=lambda e: (e.get("inicio_seconds", 0), e.get("tipo_evento", "")))
    unique, seen = [], set()
    for event in events:
        key = (event.get("tipo_evento"), event.get("inicio_frame"), event.get("fin_frame"), event.get("jugador_id"), event.get("pasador_id"), event.get("receptor_id"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return unique


def resumir_secuencias_ataque(events):
    summaries = []
    for event in events or []:
        if event.get("tipo_evento") != "fin_ataque":
            continue
        start = float(event.get("inicio_seconds", 0.0))
        end = float(event.get("fin_seconds", start))
        mins, secs = int(start // 60), int(start % 60)
        team = event.get("equipo", TEAM_NEUTRAL)
        players = event.get("jugadores", [])
        summaries.append(
            f"{mins:02d}:{secs:02d}–{int(end//60):02d}:{int(end%60):02d} | equipo {team} | "
            f"jugadores {players} | pases estimados {event.get('cantidad_pases', 0)} | "
            f"conducción colectiva estimada {event.get('metros_conduccion_equipo', 0)} m"
        )
    return summaries


def distancia_jugador_en_intervalo(tracks, jugador_id, inicio_frame, fin_frame, fps=25.0, view_transformer=None):
    players_by_frame = tracks.get("players", [])
    if not players_by_frame:
        return None, "sin datos"
    start = max(0, int(inicio_frame))
    end = min(len(players_by_frame) - 1, int(fin_frame))
    if end <= start:
        return 0.0, "sin desplazamiento medible"

    previous_point = None
    previous_frame = None
    total_distance = 0.0
    calibrated_steps = 0
    estimated_steps = 0
    for frame_num in range(start, end + 1):
        player = (players_by_frame[frame_num] or {}).get(jugador_id)
        if not player:
            continue
        raw_point = _player_point(player)
        if raw_point is None:
            continue
        metric_point = _transform(view_transformer, raw_point, frame_num)
        method = "calibrado"
        if metric_point is None:
            box = _bbox(player)
            if box is None:
                previous_point = None
                previous_frame = None
                continue
            height = float(box[3] - box[1])
            if height <= 1:
                previous_point = None
                previous_frame = None
                continue
            metric_point = raw_point * (ALTURA_JUGADOR_M / height)
            method = "estimado"

        if previous_point is not None and previous_frame is not None:
            gap = frame_num - previous_frame
            if 0 < gap <= max(1, int(0.4 * fps)):
                step = _distance(metric_point, previous_point)
                max_step = (gap / fps) * MAX_SPEED_M_S
                if np.isfinite(step) and 0 <= step <= max_step:
                    total_distance += step
                    if method == "calibrado":
                        calibrated_steps += 1
                    else:
                        estimated_steps += 1
        previous_point = metric_point
        previous_frame = frame_num

    if calibrated_steps and not estimated_steps:
        method_label = "metros (calibración de cancha)"
    elif calibrated_steps and estimated_steps:
        method_label = "metros mixtos: calibración + estimación"
    elif estimated_steps:
        method_label = "metros aproximados (sin calibración fiable)"
    else:
        return None, "no se pudo medir la trayectoria"
    return round(total_distance, 2), method_label


def metricas_jugador_con_sin_pelota(tracks, jugador_id, inicio_frame, fin_frame, fps=25.0, view_transformer=None):
    players_by_frame = tracks.get("players", [])
    balls_by_frame = tracks.get("ball", [])
    if not players_by_frame or not balls_by_frame:
        return {"total_m": None, "con_pelota_m": None, "sin_pelota_m": None, "metodo": "sin datos"}
    start = max(0, int(inicio_frame))
    end = min(len(players_by_frame), len(balls_by_frame)) - 1
    end = min(end, int(fin_frame))
    if end <= start:
        return {"total_m": 0.0, "con_pelota_m": 0.0, "sin_pelota_m": 0.0, "metodo": "sin desplazamiento medible"}

    total_m = with_ball = without_ball = 0.0
    calibrated = estimated = 0
    prev_point = prev_frame = prev_has_ball = None
    for frame_num in range(start, end + 1):
        players = players_by_frame[frame_num] or {}
        player = players.get(jugador_id)
        if not player:
            prev_point = prev_frame = prev_has_ball = None
            continue
        raw = _player_point(player)
        if raw is None:
            prev_point = prev_frame = prev_has_ball = None
            continue
        metric = _transform(view_transformer, raw, frame_num)
        method = "calibrado"
        if metric is None:
            box = _bbox(player)
            if box is None or float(box[3] - box[1]) <= 1:
                prev_point = prev_frame = prev_has_ball = None
                continue
            metric = raw * (ALTURA_JUGADOR_M / float(box[3] - box[1]))
            method = "estimado"
        ball = (balls_by_frame[frame_num] or {}).get(1, {})
        bp = _ball_point(ball)
        owner_info = _owner(players, bp)
        has_ball = bool(owner_info and owner_info[0] == jugador_id)
        if prev_point is not None and prev_frame is not None:
            gap = frame_num - prev_frame
            if 0 < gap <= max(1, int(0.4 * fps)):
                step = _distance(metric, prev_point)
                max_step = (gap / fps) * MAX_SPEED_M_S
                if np.isfinite(step) and 0 <= step <= max_step:
                    total_m += step
                    if has_ball or prev_has_ball:
                        with_ball += step
                    else:
                        without_ball += step
                    if method == "calibrado":
                        calibrated += 1
                    else:
                        estimated += 1
        prev_point, prev_frame, prev_has_ball = metric, frame_num, has_ball

    if calibrated and estimated:
        label = "mixto: calibración + estimación"
    elif calibrated:
        label = "calibrado en cancha"
    elif estimated:
        label = "aproximado; cancha sin calibración fiable"
    else:
        label = "sin desplazamiento medible"
    return {"total_m": round(total_m, 2), "con_pelota_m": round(with_ball, 2),
            "sin_pelota_m": round(without_ball, 2), "metodo": label}


def _poseedor_en_frame(players, ball_point, max_factor=1.65):
    if ball_point is None:
        return None
    choices = []
    for tid, player in players.items():
        if player.get("role", "field_player") not in ("field_player", "goalkeeper"):
            continue
        team = player.get("team", TEAM_NEUTRAL)
        if team not in (TEAM_ARGENTINA, TEAM_ENGLAND):
            continue
        point = _player_point(player)
        box = _bbox(player)
        if point is None or box is None:
            continue
        d = _distance(point, ball_point)
        h = max(1.0, float(box[3] - box[1]))
        choices.append((d / h, d, tid))
    if not choices:
        return None
    ratio, _, tid = min(choices, key=lambda x: x[0])
    return tid if ratio <= max_factor else None


def _primer_frame_posesion_del_autor(tracks, autor_id, frame_remate, fps, ventana_previa_s):
    ids_del_autor = _unificar_trayectorias_y_equipos(tracks, autor_id, frame_remate, ventana_previa_s, fps)
    
    players_by_frame = tracks.get("players", [])
    balls_by_frame = tracks.get("ball", [])
    total = min(len(players_by_frame), len(balls_by_frame))
    if total == 0:
        return max(0, frame_remate)

    limite = max(0, frame_remate - int(ventana_previa_s * fps))
    primer_frame_posesion = None
    
    for f in range(frame_remate, limite - 1, -1):
        players = players_by_frame[f] or {}
        ball = (balls_by_frame[f] or {}).get(1, {})
        ball_px = _ball_point(ball)
        if ball_px is None:
            continue
            
        poseedor_id = _poseedor_en_frame(players, ball_px)
        
        if poseedor_id in ids_del_autor:
            primer_frame_posesion = f
            
    if primer_frame_posesion is not None:
        return primer_frame_posesion
        
    return limite


def distancia_autor_gol(tracks, events, fps=25.0, view_transformer=None, ventana_previa_s=60.0):
    players_by_frame = tracks.get("players", [])
    if not players_by_frame or not events:
        return []
    total_frames = len(players_by_frame)
    ordenados = sorted(events, key=lambda e: float(e.get("inicio_seconds", 0)))

    resultados = []
    for evento in ordenados:
        if evento.get("tipo_evento") != "tiro_al_arco":
            continue
        tirador = evento.get("tirador_id")
        if tirador in (None, ""):
            continue
        fin_frame = min(total_frames - 1, int(evento.get("fin_frame", evento.get("inicio_frame", 0))))
        frame_remate = int(evento.get("inicio_frame", fin_frame))
        team = evento.get("equipo", TEAM_NEUTRAL)

        inicio_frame = _primer_frame_posesion_del_autor(
            tracks, tirador, frame_remate, fps, ventana_previa_s
        )
        
        ids_del_autor = _unificar_trayectorias_y_equipos(tracks, tirador, frame_remate, ventana_previa_s, fps)
        
        metros_totales = 0.0
        metodos = []
        
        for un_id in ids_del_autor:
            m, met = distancia_jugador_en_intervalo(
                tracks, un_id, inicio_frame, fin_frame, fps=fps, view_transformer=view_transformer
            )
            if m is not None:
                metros_totales += m
                metodos.append(met)

        metodo_final = metodos[0] if metodos else "sin datos"

        ventana_inicio = max(0, frame_remate - int(ventana_previa_s * fps))
        pases_previos = [
            e for e in ordenados
            if e is not evento
            and e.get("tipo_evento") == "pase_completado_posible"
            and e.get("equipo", TEAM_NEUTRAL) in (team, TEAM_NEUTRAL)
            and ventana_inicio <= int(e.get("inicio_frame", 0)) <= frame_remate
        ]
        jugadores_previos = sorted({
            j for e in ordenados
            if e is not evento
            and e.get("equipo", TEAM_NEUTRAL) in (team, TEAM_NEUTRAL)
            and ventana_inicio <= int(e.get("inicio_frame", 0)) <= frame_remate
            for j in (e.get("jugador_id"), e.get("pasador_id"), e.get("receptor_id"))
            if j not in (None, "")
        })

        resultados.append({
            "tirador_id": tirador,
            "inicio_frame": inicio_frame,
            "fin_frame": fin_frame,
            "inicio_seconds": round(inicio_frame / fps, 3),
            "fin_seconds": round(fin_frame / fps, 3),
            "metros_autor": round(metros_totales, 2),
            "metodo": metodo_final,
            "pases_previos": len(pases_previos),
            "jugadores_previos": jugadores_previos,
            "equipo": team,
        })
    return resultados