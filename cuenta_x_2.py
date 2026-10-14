# ============================================================
# descargar_videos.py
# Descarga videos de YouTube, X (Twitter) y otros sitios
# soportados por yt-dlp, en la máxima calidad disponible.
# ============================================================

import sys
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp


# Carpeta donde se guardan los videos
CARPETA_SALIDA = Path.home() / "Downloads" / "VideosYouTube_HD"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Detección del sitio según la URL
# ------------------------------------------------------------

def detectar_sitio(url: str) -> str:
    """
    Devuelve un identificador del sitio a partir de la URL.
    Valores posibles: 'youtube', 'x', 'otro'.
    """
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return "otro"

    # Quitar 'www.' y puertos
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]

    if host in ("youtube.com", "youtu.be", "m.youtube.com", "music.youtube.com"):
        return "youtube"

    if host in ("x.com", "twitter.com", "mobile.twitter.com", "mobile.x.com"):
        return "x"

    return "otro"


# ------------------------------------------------------------
# Opciones por sitio
# ------------------------------------------------------------

def opciones_youtube() -> dict:
    """Configuración óptima para YouTube (mejor calidad + mp4)."""
    return {
        # Baja la mejor pista de video y la mejor de audio por separado,
        # luego yt-dlp las fusiona automáticamente con ffmpeg.
        "format": "bestvideo+bestaudio/best",

        # Orden de preferencia para la pista de video:
        # 1) máxima resolución
        # 2) máximo framerate
        # 3) mejor códec (av01 > vp9 > h264)
        "format_sort": ["res", "fps", "vcodec"],

        # Forzar contenedor mp4
        "merge_output_format": "mp4",

        # Nombre: titulo-del-video-ID.mp4
        "outtmpl": str(CARPETA_SALIDA / "%(title).120s-%(id)s.%(ext)s"),

        # Comportamiento
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 5,

        # Post-procesamiento: forzar mp4
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }


def opciones_x() -> dict:
    """Configuración óptima para X (Twitter)."""
    return {
        "format": "best[ext=mp4]/best",
        "outtmpl": str(CARPETA_SALIDA / "x-%(uploader_id)s-%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "retries": 10,
        "fragment_retries": 10,

        # --- ESTA ES LA CLAVE ---
        # Lee las cookies del navegador donde tengas la sesión de X iniciada
        # Opciones: "chrome", "firefox", "edge", "brave", "opera", "vivaldi", "safari"
        "cookiesfrombrowser": ("firefox",),  # <-- Cambia a tu navegador principal

        # Opcional: simular un User-Agent de navegador real ayuda a evitar bloqueos
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },

        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }

def opciones_genericas() -> dict:
    """Configuración genérica para cualquier otro sitio soportado."""
    return {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": str(CARPETA_SALIDA / "%(title).120s-%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 5,
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }


# ------------------------------------------------------------
# Función principal
# ------------------------------------------------------------

def descargar_video(url: str):
    """
    Detecta el sitio y descarga el video en la mejor calidad disponible.
    Combina la mejor pista de video + la mejor de audio cuando aplica.
    """
    sitio = detectar_sitio(url)

    if sitio == "youtube":
        print("Sitio detectado: YouTube")
        opciones = opciones_youtube()
    elif sitio == "x":
        print("Sitio detectado: X (Twitter)")
        opciones = opciones_x()
    else:
        print("Sitio no identificado. Usando configuración genérica.")
        opciones = opciones_genericas()

    with yt_dlp.YoutubeDL(opciones) as ydl:
        info = ydl.extract_info(url, download=True)

    # Mostrar resumen de lo descargado
    print("\n" + "=" * 60)
    print("DESCARGA COMPLETA")
    print("=" * 60)
    print(f"Título:      {info.get('title')}")
    print(f"Canal:       {info.get('uploader')}")
    print(f"Duración:    {info.get('duration')} segundos")
    print(f"Resolución:  {info.get('width')}x{info.get('height')}")
    print(f"FPS:         {info.get('fps')}")
    print(f"Códec:       {info.get('vcodec')}")
    print(f"Carpeta:     {CARPETA_SALIDA}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    print("=== DESCARGADOR DE VIDEOS (YouTube / X / otros) ===\n")

    url = input("Pegá el enlace del video: ").strip()

    if not url:
        print("No ingresaste ningún enlace.")
        sys.exit(1)

    try:
        descargar_video(url)
    except Exception as error:
        print(f"\nError al descargar:\n{error}")

    input("\nPresioná Enter para salir...")