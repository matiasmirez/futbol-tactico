"""Detector robusto de marcas parciales de cancha para vídeo de fútbol.

Detecta segmentos blancos y ajusta rectas para buscar intersecciones extrapoladas.
No presupone que se vean las cuatro esquinas. La detección de líneas, por sí sola,
no basta para asignar metros: la homografía solo debe aceptarse cuando las marcas
se puedan asociar a una geometría conocida de la cancha.
"""
from __future__ import annotations

import cv2
import numpy as np


def detectar_segmentos_cancha(frame):
    """Detecta líneas blancas dentro de la superficie verde, no en tribunas/carteles.

    La máscara de césped actúa como restricción espacial: los píxeles blancos solo
    se consideran candidatos si están cerca de una zona reconocida como césped.
    Esto reduce marcadamente los bordes del marcador, publicidad y gradas. No
    asigna metros ni crea una homografía por sí solo.
    """
    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return []
    if len(frame.shape) == 2:
        gray = frame
        hsv = None
        green_near = np.ones_like(gray, dtype=np.uint8) * 255
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Césped natural/artificial: tono verde, con saturación mínima para
        # no confundir superficies grises/blancas con césped.
        grass = cv2.inRange(hsv, np.array([24, 28, 25], np.uint8),
                            np.array([105, 255, 255], np.uint8))
        # Cerrar huecos por sombras y textura. Dilatar permite conservar la
        # línea blanca aunque su propio píxel no sea verde.
        grass = cv2.morphologyEx(
            grass, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        green_near = cv2.dilate(
            grass, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
            iterations=1,
        )
    else:
        return []

    if hsv is not None:
        # Blanco brillante con saturación baja.
        white = cv2.inRange(hsv, np.array([0, 0, 165], dtype=np.uint8),
                            np.array([180, 105, 255], dtype=np.uint8))
        # Un borde blanco solo cuenta si hay césped en sus inmediaciones.
        mask = cv2.bitwise_and(white, green_near)
    else:
        _, mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    edges = cv2.Canny(mask, 45, 130)
    h, w = mask.shape[:2]
    min_len = max(35, int(min(w, h) * 0.045))
    raw = cv2.HoughLinesP(
        edges, 1, np.pi / 720,
        threshold=max(35, int(min(w, h) * 0.045)),
        minLineLength=min_len,
        maxLineGap=max(14, int(min(w, h) * 0.025)),
    )
    if raw is None:
        return []

    try:
        rows = np.asarray(raw).reshape(-1, 4)
    except (ValueError, TypeError):
        return []

    segments = []
    for row in rows:
        x1, y1, x2, y2 = (int(v) for v in row)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue
        # Comprobar soporte blanco/césped a lo largo del segmento, no solo sus
        # extremos: elimina muchos bordes de camisetas y ruido aislado.
        samples = max(8, min(40, int(length / 12)))
        xs = np.clip(np.linspace(x1, x2, samples).astype(int), 0, w - 1)
        ys = np.clip(np.linspace(y1, y2, samples).astype(int), 0, h - 1)
        grass_support = float(np.mean(green_near[ys, xs] > 0))
        white_support = float(np.mean(mask[ys, xs] > 0))
        if hsv is not None and grass_support < 0.65:
            continue
        if white_support < 0.12:
            continue
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
        segments.append({
            "p1": (x1, y1),
            "p2": (x2, y2),
            "length_px": length,
            "angle_deg": angle,
            "grass_support": grass_support,
            "white_support": white_support,
        })
    return sorted(segments, key=lambda item: item["length_px"], reverse=True)


def ajustar_recta(segmentos):
    """Ajusta ax+by+c=0 usando los extremos de segmentos válidos."""
    if not segmentos:
        return None
    pts = []
    for segment in segmentos:
        if not isinstance(segment, dict):
            continue
        p1, p2 = segment.get("p1"), segment.get("p2")
        if p1 is None or p2 is None:
            continue
        try:
            a = np.asarray(p1, dtype=np.float32).reshape(-1)
            b = np.asarray(p2, dtype=np.float32).reshape(-1)
            if a.size >= 2 and b.size >= 2 and np.isfinite(a[:2]).all() and np.isfinite(b[:2]).all():
                pts.extend([a[:2], b[:2]])
        except (TypeError, ValueError):
            continue
    if len(pts) < 2:
        return None
    points = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    try:
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    except (cv2.error, ValueError):
        return None
    a, b = float(-vy), float(vx)
    c = -(a * float(x0) + b * float(y0))
    norm = float(np.hypot(a, b))
    if not np.isfinite([a, b, c, norm]).all() or norm < 1e-8:
        return None
    return (a / norm, b / norm, c / norm)


def interseccion_rectas(r1, r2, eps=1e-5):
    """Intersección de rectas ax+by+c=0; None si son casi paralelas."""
    if r1 is None or r2 is None:
        return None
    try:
        a1, b1, c1 = (float(v) for v in r1)
        a2, b2, c2 = (float(v) for v in r2)
    except (TypeError, ValueError):
        return None
    det = a1 * b2 - a2 * b1
    if not np.isfinite(det) or abs(det) < eps:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    if not np.isfinite([x, y]).all():
        return None
    return (float(x), float(y))


def crear_diagnostico(frame, salida):
    """Guarda una imagen de diagnóstico con los segmentos detectados."""
    segments = detectar_segmentos_cancha(frame)
    if frame is None or not hasattr(frame, "copy"):
        return segments
    vis = frame.copy()
    for segment in segments:
        cv2.line(vis, segment["p1"], segment["p2"], (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(salida), vis)
    return segments
