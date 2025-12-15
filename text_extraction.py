from paddleocr import PaddleOCR
from pathlib import Path
import json
import re

# =====================================================
# 0) SETTINGS
# =====================================================
IMAGES_DIR = Path("images")   # dossier racine (avec sous-dossiers pays)
BASE_IMAGE = None             # sera choisie automatiquement
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================
# 1) OCR INIT
# =====================================================
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="fr",
    show_log=False
)

# =====================================================
# 2) UTILS
# =====================================================
DATE_REGEX = re.compile(r"\d{2}[./]\d{2}[./]\d{4}")

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

    try:
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
    except Exception:
        return None


def is_name(t):
    return t.isupper() and t.isalpha() and 3 <= len(t) <= 25


def is_city(t):
    return t.isupper() and len(t) >= 5 and not any(c.isdigit() for c in t)


# =====================================================
# 3) PROCESS ONE IMAGE
# =====================================================
def process_image(img_path: Path):
    print(f"\n📄 Processing: {img_path}")

    results = ocr.ocr(str(img_path))
    if not results or not results[0]:
        return None

    # OCR blocks top → bottom
    blocks = []
    for box, (text, conf) in results[0]:
        y = (box[0][1] + box[2][1]) / 2
        blocks.append((y, text.strip()))

    blocks = [t for _, t in sorted(blocks)]

    # ---------------- MRZ ----------------
    mrz_lines = [b for b in blocks if "<" in b and len(b) >= 40]
    mrz_data = parse_mrz(mrz_lines)

    # ---------------- Dates ----------------
    dates = []
    for b in blocks:
        if DATE_REGEX.fullmatch(b):
            dates.append(b.replace(".", "").replace("/", ""))
    dates = sorted(dates)

    # ---------------- Names ----------------
    names = [b for b in blocks if is_name(b)]

    # ---------------- Cities ----------------
    cities = [b for b in blocks if is_city(b)]

    # ---------------- Final merge ----------------
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

    return structured


# =====================================================
# 4) MAIN – BASE IMAGE + GENERALISATION
# =====================================================
all_images = list(IMAGES_DIR.rglob("*.jpg"))

assert all_images, "❌ No images found"

BASE_IMAGE = all_images[0]   # image de référence

print(f"\n🧩 Base image selected: {BASE_IMAGE.name}")

for img in all_images:
    data = process_image(img)
    if data:
        out = OUTPUT_DIR / f"{img.stem}_structured.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"✅ Saved → {out.name}")
