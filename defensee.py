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
    # Preferimos el punto de contacto con el suelo para jugadores.
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
    # Solo se reportan metros reales si el punto cae dentro de una calibración válida.
    transformed = _transform(transformer, raw_point, frame_num)
    if transformed is not None:
        return transformed, "m"
    box = _bbox(player)
    if box is None:
        return None, "px"
    height = float(box[3] - box[1])
    if height <= 1:
        return None, "px"
    # Fallback aproximado por altura aparente; perspectiva puede distorsionar mucho.
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


def detectar_alertas_defensivas(
    tracks, fps=25.0, salida_csv=None, tiempos_goles=None,
    equipo_en_posesion=None, view_transformer=None, bbox_arco=None,
    **kwargs,
):
    """Devuelve eventos temporales compatibles con el panel del video.

    `tiempos_goles` y `salida_csv` se aceptan por compatibilidad, pero no se usan:
    la reconstrucción es automática y no depende de un CSV ni de marcas manuales.
    """
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
        # Posición del balón en metros cuando la calibración está disponible.
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
        # Desplazamiento del balón entre observaciones; limita saltos por oclusiones.
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
                # Cambio de poseedor del mismo equipo: posible pase/recepción.
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
                        # En píxeles la métrica solo se escala si no existe homografía; la etiqueta aclara aproximación.
                        transformed_now = _transform(view_transformer, player_point, i)
                        transformed_prev = _transform(view_transformer, carry_last_point, carry_last_frame)
                        if transformed_now is not None and transformed_prev is not None:
                            step = _distance(transformed_now, transformed_prev)
                        else:
                            box = _bbox(owner)
                            h = float(box[3] - box[1]) if box is not None else 0.0
                            step = _distance(player_point, carry_last_point) * (ALTURA_JUGADOR_M / h) if h > 1 else 0.0
                        # Evitar sumar saltos imposibles por ID tracking inestable.
                        if np.isfinite(step) and 0 <= step <= 4.0:
                            carry_distance += step
                            team_carry_total[team] += step
                    carry_last_point, carry_last_frame = player_point.copy(), i

            # Candidato de remate: pelota muy rápida y poseedor cercano al balón.
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
            # Ball moving quickly while owner is unknown can be a shot, pass, clearance or occlusion.
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

    # Cerrar conducción y ataque al final del clip.
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

    # Marcar gol solo como probable cuando el balón desaparece tras una trayectoria rápida cerca del borde
    # de imagen. No se afirma gol confirmado sin geometría de arco / línea de gol.
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
        p = row["ball_px"]
        # Desaparición cerca de borde de imagen: solo candidato a revisar, muy propenso a oclusión.
        # Se deja deliberadamente conservador y no se etiqueta como gol confirmado.
        # No se crea evento si no hay señal de movimiento hacia el borde.
        # Los datos del video no contienen necesariamente la línea del arco.
        # Por ello se omite inferir un gol a partir de una mera desaparición.
        pass

    # Orden temporal y eliminación de duplicados exactos.
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
    """Produce resúmenes legibles para la consola; no escribe archivos."""
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
    """Estima la distancia recorrida por un track en un intervalo de fotogramas.

    Devuelve (metros, metodo). Si no hay calibración de cancha, la distancia
    se aproxima usando la altura aparente del jugador y no debe tratarse como exacta.
    """
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
                if np.isfinite(step) and 0 <= step <= 4.0:
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
    """Estima distancia total, con pelota y sin pelota para un track en un intervalo.

    La posesión se aproxima por cercanía jugador-balón. Sin calibración válida,
    la distancia usa una escala basada en la altura aparente y queda marcada como estimada.
    Devuelve dict con metros y etiqueta de método.
    """
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
                if np.isfinite(step) and 0 <= step <= 4.0:
                    total_m += step
                    # Clasifica el tramo como con pelota si la posesión se estima en alguno de sus extremos.
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


def distancia_autor_gol(
    tracks, events, fps=25.0, view_transformer=None, ventana_previa_segundos=15.0
):
    """Estima la distancia recorrida por el jugador asociado a un posible remate.

    El tracking disponible no permite confirmar por sí solo que un remate haya
    terminado en gol. Por compatibilidad con el informe existente, el nombre
    histórico de la función se conserva, pero devuelve candidatos a autor de
    remate (no goles confirmados).

    Cada resultado contiene la distancia del jugador en la ventana previa al
    evento, el método de medición, pases inferidos y jugadores involucrados.
    """
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    players_by_frame = tracks.get("players", [])
    if not players_by_frame:
        return []

    total_frames = len(players_by_frame)
    shot_events = [
        event for event in (events or [])
        if event.get("tipo_evento") == "tiro_al_arco"
        and event.get("tirador_id") not in (None, "")
    ]
    pass_events = [
        event for event in (events or [])
        if event.get("tipo_evento") == "pase_completado_posible"
    ]

    results = []
    seen = set()
    window_frames = max(1, int(max(0.0, float(ventana_previa_segundos)) * fps))

    for shot in sorted(shot_events, key=lambda e: int(e.get("inicio_frame", 0))):
        try:
            shooter_id = int(shot.get("tirador_id"))
            shot_frame = max(0, min(total_frames - 1, int(shot.get("inicio_frame", 0))))
            team = shot.get("equipo", TEAM_NEUTRAL)
        except (TypeError, ValueError):
            continue

        # Evita repetir el mismo tirador cuando el detector genera varios
        # candidatos de remate muy cercanos en el tiempo.
        dedup_key = (shooter_id, shot_frame // max(1, int(2 * fps)))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        start_frame = max(0, shot_frame - window_frames)
        end_frame = max(start_frame, min(total_frames - 1, int(shot.get("fin_frame", shot_frame))))

        distance, method = distancia_jugador_en_intervalo(
            tracks, shooter_id, start_frame, end_frame,
            fps=fps, view_transformer=view_transformer
        )

        start_seconds = start_frame / fps
        end_seconds = end_frame / fps
        prior_passes = [
            event for event in pass_events
            if start_seconds <= float(event.get("inicio_seconds", -1)) <= end_seconds
            and event.get("equipo", team) == team
        ]

        involved = []
        for event in prior_passes:
            for key in ("pasador_id", "receptor_id"):
                player_id = event.get(key)
                if player_id not in (None, "") and player_id not in involved:
                    involved.append(player_id)

        if shooter_id not in involved:
            involved.append(shooter_id)

        results.append({
            "tipo_evento": "posible_remate",
            "tirador_id": shooter_id,
            "equipo": team,
            "inicio_seconds": round(start_frame / fps, 3),
            "fin_seconds": round(end_frame / fps, 3),
            "inicio_frame": start_frame,
            "fin_frame": end_frame,
            "metros_autor": distance,
            "metodo": method,
            "pases_previos": len(prior_passes),
            "jugadores_previos": involved,
        })

    return results
