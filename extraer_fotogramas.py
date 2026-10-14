import cv2
from pathlib import Path

video = Path(r"C:\Users\Matias\Downloads\chivas_penal.mp4")
salida = Path("fotogramas_calibracion")
salida.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise SystemExit(f"No se pudo abrir el video: {video}")

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Fotogramas: {total} | FPS: {fps:.2f}")

if total <= 0:
    raise SystemExit("El video no contiene fotogramas legibles.")

cantidad = min(12, total)
indices = sorted(set(round(i * (total - 1) / max(1, cantidad - 1))
                     for i in range(cantidad)))

for n in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, frame = cap.read()
    if ok:
        nombre = salida / f"frame_{n:06d}.jpg"
        cv2.imwrite(str(nombre), frame)
        print(f"Guardado: {nombre} | segundo {n / fps:.2f}")

cap.release()
print(f"\nListo. Revisá las imágenes en: {salida.resolve()}")
