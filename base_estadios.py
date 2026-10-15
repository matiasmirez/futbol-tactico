"""Lectura segura de estadios.json y dimensiones reglamentarias."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "estadios.json"

def cargar_base_estadios(path=DEFAULT_DB):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("estadios"), list):
            raise ValueError("Formato inválido: se espera un objeto con 'estadios'.")
        return data
    except FileNotFoundError:
        return {"version": 1, "reglas_ifab": {}, "estadios": []}

def buscar_estadio(query: str, path=DEFAULT_DB) -> Optional[dict]:
    """Busca por id, nombre, ciudad o país, sin asumir coincidencias ambiguas."""
    if not query or not query.strip():
        return None
    q = query.casefold().strip()
    data = cargar_base_estadios(path)
    exactos = []
    parciales = []
    for s in data["estadios"]:
        campos = [str(s.get(k, "")) for k in ("id", "nombre", "ciudad", "pais")]
        normalizados = [v.casefold().strip() for v in campos]
        if q in normalizados:
            exactos.append(s)
        elif any(q in v for v in normalizados):
            parciales.append(s)
    resultados = exactos or parciales
    return resultados[0] if len(resultados) == 1 else None

def dimensiones_campo(query: str = "", path=DEFAULT_DB):
    """
    Devuelve dimensiones verificadas si hay un registro exacto/no ambiguo.
    Si no, devuelve None: NO inventa dimensiones del estadio.
    """
    s = buscar_estadio(query, path) if query else None
    if not s:
        return None
    largo, ancho = s.get("largo_m"), s.get("ancho_m")
    if not isinstance(largo, (int, float)) or not isinstance(ancho, (int, float)):
        return None
    if not (90 <= float(largo) <= 120 and 45 <= float(ancho) <= 90):
        return None
    if s.get("estado_verificacion") != "verificado":
        return None
    return float(largo), float(ancho)
