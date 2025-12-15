from paddleocr import PaddleOCR
from pathlib import Path
import json
import re

# =====================================================
# 1) OCR
# =====================================================
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="fr",
    show_log=False
)

img_path = Path("document_id/images/000013.jpg")
assert img_path.exists(), "❌ Image introuvable"

results = ocr.ocr(str(img_path))
assert results and results[0], "❌ OCR vide"

print("✅ OCR terminé")

# =====================================================
# 2) COLLECT & SORT BLOCKS (TOP → BOTTOM)
# =====================================================
blocks = []
for box, (text, conf) in results[0]:
    y_center = (box[0][1] + box[2][1]) / 2
    blocks.append(text.strip())

# =====================================================
# 3) MRZ DETECTION
# =====================================================
mrz_lines = [b for b in blocks if "<" in b and len(b) >= 40]

# =====================================================
# 4) ICAO CHECKSUM
# =====================================================
def mrz_checksum(s):
    weights = [7, 3, 1]
    total = 0
    for i, c in enumerate(s):
        if c.isdigit():
            v = int(c)
        elif c == "<":
            v = 0
        else:
            v = ord(c) - 55
        total += v * weights[i % 3]
    return total % 10

def parse_mrz(lines):
    if len(lines) < 2:
        return None

    l1, l2 = lines[0], lines[1]
    if len(l1) != 44 or len(l2) != 44:
        return None

    if mrz_checksum(l2[:9]) != int(l2[9]):
        return None

    last, first = l1[5:].split("<<", 1)
    return {
        "last_name": last.replace("<", ""),
        "first_name": first.replace("<", ""),
        "passport_number": l2[:9],
        "nationality": l2[10:13],
        "date_of_birth": l2[13:19],
        "date_of_expiry": l2[21:27]
    }

mrz_data = parse_mrz(mrz_lines)

# =====================================================
# 5) FALLBACK EXTRACTION
# =====================================================
DATE_REGEX = re.compile(r"\d{2}[./]\d{2}[./]\d{4}")

dates = []
for b in blocks:
    if DATE_REGEX.fullmatch(b):
        dates.append(b.replace(".", "").replace("/", ""))

dates = sorted(dates)

def is_name(t):
    return t.isupper() and t.isalpha() and 3 <= len(t) <= 20

names = [b for b in blocks if is_name(b)]

def is_city(t):
    return t.isupper() and len(t) >= 6 and not any(c.isdigit() for c in t)

cities = [b for b in blocks if is_city(b)]

# =====================================================
# 6) FINAL MERGE
# =====================================================
structured = {
    "last_name": mrz_data["last_name"] if mrz_data else (names[0] if len(names) >= 1 else None),
    "first_name": mrz_data["first_name"] if mrz_data else (names[1] if len(names) >= 2 else None),
    "passport_number": mrz_data["passport_number"] if mrz_data else None,
    "nationality": mrz_data["nationality"] if mrz_data else None,
    "date_of_birth": mrz_data["date_of_birth"] if mrz_data else (dates[0] if len(dates) >= 1 else None),
    "date_of_issue": dates[1] if len(dates) >= 2 else None,
    "date_of_expiry": mrz_data["date_of_expiry"] if mrz_data else (dates[-1] if len(dates) >= 3 else None),
    "city_of_issue": cities[0] if cities else None
}

# =====================================================
# 7) OUTPUT
# =====================================================
print("\n📦 DONNÉES STRUCTURÉES :\n")
print(json.dumps(structured, indent=4, ensure_ascii=False))

output_json = img_path.parent / f"{img_path.stem}_structured.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(structured, f, indent=4, ensure_ascii=False)

print("\n✅ JSON sauvegardé :", output_json)
