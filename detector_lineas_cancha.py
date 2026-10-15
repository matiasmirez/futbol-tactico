"""Detector de marcas parciales de cancha para vídeo de fútbol.

No presupone que se vean las cuatro esquinas del área. Detecta segmentos blancos,
ajusta rectas y calcula intersecciones extrapoladas. No convierte automáticamente
esas intersecciones a metros: para eso hace falta asociar al menos cuatro puntos
no colineales con coordenadas reales de cancha.
"""
from __future__ import annotations
import cv2
import numpy as np


def detectar_segmentos_cancha(frame):
    """Devuelve segmentos probables de líneas blancas del campo (x1,y1,x2,y2)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Blanco poco saturado; restringir por brillo para reducir jugadores/camisetas.
    mask = cv2.inRange(hsv, np.array([0, 0, 155]), np.array([180, 95, 255]))
    # Quitar ruido pequeño sin borrar líneas finas.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edges = cv2.Canny(mask, 45, 130)
    h, w = frame.shape[:2]
    raw = cv2.HoughLinesP(edges, 1, np.pi / 720,
                           threshold=max(35, int(min(w, h) * .045)),
                           minLineLength=max(35, int(min(w, h) * .045)),
                           maxLineGap=max(12, int(min(w, h) * .018)))
    if raw is None:
        return []
    segments = []
    for x1, y1, x2, y2 in raw[:, 0]:
        length = float(np.hypot(x2-x1, y2-y1))
        angle = float(np.degrees(np.arctan2(y2-y1, x2-x1)) % 180)
        # Ignorar segmentos casi puntuales y diagonales muy cortas.
        if length >= max(35, min(w, h) * .045):
            segments.append({"p1": (int(x1), int(y1)), "p2": (int(x2), int(y2)),
                             "length_px": length, "angle_deg": angle})
    return sorted(segments, key=lambda s: s["length_px"], reverse=True)


def ajustar_recta(segmentos):
    """Ajusta una recta ax+by+c=0 a los puntos de varios segmentos."""
    pts = []
    for s in segmentos:
        pts.extend([s["p1"], s["p2"]])
    if len(pts) < 2:
        return None
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    a, b = float(-vy), float(vx)
    c = -(a * float(x0) + b * float(y0))
    norm = np.hypot(a, b)
    if norm < 1e-8:
        return None
    return (a/norm, b/norm, c/norm)


def interseccion_rectas(r1, r2, eps=1e-5):
    """Intersección de dos rectas; None si son paralelas/casi paralelas."""
    if r1 is None or r2 is None:
        return None
    a1,b1,c1 = r1
    a2,b2,c2 = r2
    det = a1*b2-a2*b1
    if abs(det) < eps:
        return None
    x = (b1*c2-b2*c1)/det
    y = (c1*a2-c2*a1)/det
    if not np.isfinite([x,y]).all():
        return None
    return (float(x), float(y))


def crear_diagnostico(frame, salida):
    """Guarda overlay para revisar las líneas parciales detectadas."""
    segs = detectar_segmentos_cancha(frame)
    vis = frame.copy()
    for s in segs:
        cv2.line(vis, s["p1"], s["p2"], (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(salida), vis)
    return segs
