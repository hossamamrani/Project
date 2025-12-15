from paddleocr import PaddleOCR
from pathlib import Path

# =========================
# Initialiser PaddleOCR
# =========================
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="fr"
)

# =========================
# Charger l'image
# =========================
img_path = Path("document_id/photo_mediapipe.png")

results = ocr.ocr(str(img_path))

# =========================
# Fichier de sortie texte
# =========================
output_path = img_path.parent / f"{img_path.stem}_ocr.txt"

# =========================
# Sauvegarde des résultats OCR
# =========================
with open(output_path, "w", encoding="utf-8") as f:
    for line in results[0]:
        box, (text, confidence) = line
        f.write(f"Zone: {box}\n")
        f.write(f"Texte détecté: {text} (confiance: {confidence:.2f})\n\n")

print(f"Résultats OCR sauvegardés dans : {output_path}")
