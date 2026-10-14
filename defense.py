"""
defensa.py
-----------
Modulo heuristico para señalar posibles situaciones de riesgo defensivo en un
video de futbol ya procesado por futbol.py.

No confirma errores arbitrales/tacticos ni detecta goles por sí solo: genera
alertas para revisar manualmente. Funciona con tracks organizados por frame,
como los que suelen producir los proyectos basados en YOLO + supervision.

Uso desde futbol.py:
    from defensa import detectar_alertas_defensivas

    alertas = detectar_alertas_defensivas(
        tracks=tracks,
        fps=fps,
        salida_csv=OUTPUT_DIR / f"defensa_{CLIP_NAME}.csv",
        tiempos_goles=[135.0, 248.5],  # segundos desde el inicio del video
        equipo_en_posesion=team_ball_control,  # opcional
    )
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable


# Umbrales relativos al alto mediano de los jugadores detectados.
# Son heurísticos y deben calibrarse para cada tipo de cámara/video.
RADIO_POSESION = 3.5
RADIO_RECEPTOR_LIBRE = 10.0
RADIO_PRESION = 3.5
RADIO_SUPERIORIDAD = 6.0
VENTANA_RELACION_GOL_SEGUNDOS = 10.0
SEPARACION_EVENTOS_SEGUNDOS = 1.5


def _es_numero(valor: Any) -> bool:
    try:
        return math.isfinite(float(valor))
    except (TypeError, ValueError):
        return False


def _bbox(obj: Any):
    """Devuelve bbox como [x1, y1, x2, y2], o None si no se puede leer."""
    if obj is None:
        return None

    # Estructura habitual: tracks[...][frame][id]["bbox"]
    if isinstance(obj, dict):
        for key in ("bbox", "xyxy", "box"):
            if key in obj:
                value = obj[key]
                try:
                    vals = list(value)
                    if len(vals) >= 4 and all(_es_numero(v) for v in vals[:4]):
                        return [float(v) for v in vals[:4]]
                except (TypeError, ValueError):
                    pass

        # Algunas estructuras guardan la detección dentro de "detection".
        for key in ("detection", "data"):
            if key in obj:
                found = _bbox(obj[key])
                if found is not None:
                    return found

    # Objeto supervision.Detections individual o parecido.
    for attr in ("xyxy", "bbox", "box"):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            try:
                # xyxy de supervision puede tener forma (1, 4).
                if hasattr(value, "tolist"):
                    value = value.tolist()
                while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
                    value = value[0]
                vals = list(value)
                if len(vals) >= 4 and all(_es_numero(v) for v in vals[:4]):
                    return [float(v) for v in vals[:4]]
            except (TypeError, ValueError):
                pass

    # Una lista/tupla de cuatro coordenadas.
    if isinstance(obj, (list, tuple)) and len(obj) >= 4:
        if all(_es_numero(v) for v in obj[:4]):
            return [float(v) for v in obj[:4]]

    return None


def _posicion(obj: Any):
    """Centro inferior del bbox, una aproximación a la posición en el campo."""
    box = _bbox(obj)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def _equipo(obj: Any):
    if not isinstance(obj, dict):
        return None
    for key in ("team", "team_id", "team_number"):
        value = obj.get(key)
        if value not in (None, 0, "0", ""):
            return value
    return None


def _items_frame(frame_data: Any):
    """Normaliza un frame con tracks a pares (id, objeto)."""
    if isinstance(frame_data, dict):
        return list(frame_data.items())

    # Compatibilidad con listas de detecciones.
    if isinstance(frame_data, (list, tuple)):
        return list(enumerate(frame_data))

    return []


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _obtener_frames(tracks: dict, clave: str):
    value = tracks.get(clave, [])
    return value if isinstance(value, (list, tuple)) else []


def _obtener_balon(ball_frame: Any):
    """Obtiene la posición del balón del frame. Soporta dicts por ID o lista."""
    candidates = _items_frame(ball_frame)
    for _, obj in candidates:
        pos = _posicion(obj)
        if pos is not None:
            return pos
    # Si el frame ya es un bbox/objeto individual.
    return _posicion(ball_frame)


def _obtener_jugadores(player_frame: Any):
    jugadores = []
    for player_id, obj in _items_frame(player_frame):
        pos = _posicion(obj)
        if pos is None:
            continue
        equipo = _equipo(obj)
        if equipo is None:
            continue
        box = _bbox(obj)
        altura = max(1.0, box[3] - box[1])
        jugadores.append({
            "id": player_id,
            "obj": obj,
            "pos": pos,
            "team": equipo,
            "height": altura,
        })
    return jugadores


def _normalizar_equipo(valor):
    if valor is None:
        return None
    try:
        return int(valor)
    except (TypeError, ValueError):
        return str(valor)


def detectar_alertas_defensivas(
    tracks: dict,
    fps: float,
    salida_csv: str | Path,
    tiempos_goles: Iterable[float] | None = None,
    equipo_en_posesion: Any = None,
) -> list[dict]:
    """
    Busca indicios heurísticos de peligro defensivo y exporta las alertas a CSV.

    Parámetros
    ----------
    tracks:
        Diccionario con tracks["players"] y tracks["ball"], organizados por frame.
        Cada jugador debería incluir bbox y team (o team_id).
    fps:
        FPS del video procesado.
    salida_csv:
        Ruta de salida del CSV.
    tiempos_goles:
        Segundos desde el inicio del clip en los que ocurren goles conocidos.
    equipo_en_posesion:
        Opcional. Lista por frame con el equipo en posesión. Si no se proporciona
        o no puede leerse, se infiere el equipo del jugador más cercano al balón.

    Devuelve
    --------
    list[dict]: alertas agrupadas, también guardadas en CSV.
    """
    if not isinstance(tracks, dict):
        raise TypeError("tracks debe ser un diccionario con las claves 'players' y 'ball'.")

    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = 25.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0

    players_by_frame = _obtener_frames(tracks, "players")
    ball_by_frame = _obtener_frames(tracks, "ball")
    n_frames = min(len(players_by_frame), len(ball_by_frame))
    goals = sorted(float(t) for t in (tiempos_goles or []) if _es_numero(t) and float(t) >= 0)

    alertas_frame = []

    for frame_idx in range(n_frames):
        jugadores = _obtener_jugadores(players_by_frame[frame_idx])
        balon = _obtener_balon(ball_by_frame[frame_idx])
        if balon is None or len(jugadores) < 2:
            continue

        alturas = [j["height"] for j in jugadores if j["height"] > 1]
        if not alturas:
            continue
        altura_mediana = sorted(alturas)[len(alturas) // 2]
        escala = max(8.0, altura_mediana)

        # Si el módulo principal aporta la posesión, se usa; si no, se infiere.
        atacante = None
        try:
            posesion_frame = equipo_en_posesion[frame_idx]
            posesion_frame = _normalizar_equipo(posesion_frame)
        except (TypeError, IndexError, KeyError):
            posesion_frame = None

        distancias = sorted(((_dist(j["pos"], balon), j) for j in jugadores), key=lambda item: item[0])
        jugador_cercano = distancias[0][1] if distancias else None
        if posesion_frame is None and jugador_cercano is not None:
            if distancias[0][0] <= RADIO_POSESION * escala:
                posesion_frame = _normalizar_equipo(jugador_cercano["team"])

        if posesion_frame is None:
            continue

        atacantes = [j for j in jugadores if _normalizar_equipo(j["team"]) == posesion_frame]
        defensores = [j for j in jugadores if _normalizar_equipo(j["team"]) != posesion_frame]
        if not atacantes or not defensores:
            continue

        portador = min(atacantes, key=lambda j: _dist(j["pos"], balon))
        distancia_portador = _dist(portador["pos"], balon)
        if distancia_portador > RADIO_POSESION * escala:
            continue

        razones = []
        puntuacion = 0
        distancia_defensor_portador = min(_dist(d["pos"], portador["pos"]) for d in defensores)

        # Atacante potencialmente libre cerca del balón.
        receptor_libre = None
        for a in atacantes:
            if a["id"] == portador["id"]:
                continue
            d_balon = _dist(a["pos"], balon)
            if d_balon > RADIO_RECEPTOR_LIBRE * escala:
                continue
            d_defensor = min(_dist(d["pos"], a["pos"]) for d in defensores)
            if d_defensor > 3.5 * escala:
                if receptor_libre is None or d_balon < receptor_libre["distancia_balon"]:
                    receptor_libre = {
                        "id": a["id"],
                        "distancia_balon": d_balon,
                        "distancia_defensor": d_defensor,
                    }

        if receptor_libre:
            razones.append(
                "posible_receptor_libre "
                f"(distancia_defensor={receptor_libre['distancia_defensor']:.1f}px)"
            )
            puntuacion += 45

        if distancia_defensor_portador > RADIO_PRESION * escala:
            razones.append(
                f"portador_con_poca_presion (defensor_mas_cercano={distancia_defensor_portador:.1f}px)"
            )
            puntuacion += 25

        atacantes_cerca = sum(
            _dist(a["pos"], balon) <= RADIO_SUPERIORIDAD * escala for a in atacantes
        )
        defensores_cerca = sum(
            _dist(d["pos"], balon) <= RADIO_SUPERIORIDAD * escala for d in defensores
        )
        if atacantes_cerca >= defensores_cerca + 2:
            razones.append(
                f"posible_inferioridad_local ({atacantes_cerca} atacantes vs {defensores_cerca} defensores)"
            )
            puntuacion += 15

        if puntuacion == 0:
            continue

        tiempo = frame_idx / fps
        puntuacion = min(100, puntuacion)
        alertas_frame.append({
            "frame": frame_idx,
            "time_seconds": round(tiempo, 3),
            "equipo_atacante": posesion_frame,
            "score": puntuacion,
            "reasons": "; ".join(razones),
            "distancia_portador_defensor_px": round(distancia_defensor_portador, 2),
        })

    # Agrupa alertas consecutivas para no llenar el CSV con un registro por frame.
    alertas = []
    for evento in alertas_frame:
        if (
            alertas
            and evento["equipo_atacante"] == alertas[-1]["equipo_atacante"]
            and evento["time_seconds"] - alertas[-1]["fin_seconds"] <= SEPARACION_EVENTOS_SEGUNDOS
        ):
            anterior = alertas[-1]
            anterior["fin_seconds"] = evento["time_seconds"]
            if evento["score"] > anterior["score"]:
                anterior["score"] = evento["score"]
                anterior["frame_pico"] = evento["frame"]
                anterior["razones"] = evento["reasons"]
                anterior["distancia_portador_defensor_px"] = evento["distancia_portador_defensor_px"]
        else:
            alertas.append({
                "inicio_seconds": evento["time_seconds"],
                "fin_seconds": evento["time_seconds"],
                "frame_inicio": evento["frame"],
                "frame_pico": evento["frame"],
                "equipo_atacante": evento["equipo_atacante"],
                "score": evento["score"],
                "razones": evento["reasons"],
                "distancia_portador_defensor_px": evento["distancia_portador_defensor_px"],
                "gol_cercano_seconds": "",
                "relacion_con_gol": "",
            })

    # Relaciona cada alerta con el gol siguiente si está dentro de la ventana.
    for alerta in alertas:
        gol_cercano = next(
            (g for g in goals if alerta["inicio_seconds"] <= g <= alerta["inicio_seconds"] + VENTANA_RELACION_GOL_SEGUNDOS),
            None,
        )
        if gol_cercano is not None:
            alerta["gol_cercano_seconds"] = gol_cercano
            alerta["relacion_con_gol"] = "revisar_alerta_antes_del_gol"

    # Incluye todos los goles conocidos, incluso si no se detectó alerta previa.
    for gol in goals:
        if not any(a["gol_cercano_seconds"] == gol for a in alertas):
            alertas.append({
                "inicio_seconds": max(0.0, gol - VENTANA_RELACION_GOL_SEGUNDOS),
                "fin_seconds": gol,
                "frame_inicio": "",
                "frame_pico": "",
                "equipo_atacante": "",
                "score": 100,
                "razones": "gol_registrado_manualmente; no_se_detecto_alerta_heuristica_relacionada",
                "distancia_portador_defensor_px": "",
                "gol_cercano_seconds": gol,
                "relacion_con_gol": "revisar_video_manualmente",
            })

    alertas.sort(key=lambda a: float(a["inicio_seconds"]))

    ruta = Path(salida_csv)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    columnas = [
        "inicio_seconds", "fin_seconds", "frame_inicio", "frame_pico",
        "equipo_atacante", "score", "razones",
        "distancia_portador_defensor_px", "gol_cercano_seconds", "relacion_con_gol",
    ]
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=columnas, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(alertas)

    print(f"[defensa.py] Alertas guardadas en: {ruta}")
    print(f"[defensa.py] Alertas heurísticas: {len(alertas_frame)} frames; eventos agrupados: {len(alertas)}")
    return alertas


# Alias por si preferís un nombre en inglés en futuras integraciones.
detect_defensive_alerts = detectar_alertas_defensivas
