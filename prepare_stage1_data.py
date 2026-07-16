"""
Build the Stage 1 (TB image classifier) manifest.

Unlike the old merged_dataset.csv -- which reused each X-ray image up to 53
times paired with unrelated synthetic patient records -- this manifest lists
each real image from TB_Chest_Radiography_Database exactly once, labeled
directly from its source folder (Tuberculosis/ = 1, Normal/ = 0).
"""

import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent
IMAGE_ROOT = BASE_DIR / "TB_Chest_Radiography_Database"
OUTPUT_CSV = BASE_DIR / "data" / "tb_image_manifest.csv"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def collect_images(folder: Path, label: int):
    rows = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append((str(path.relative_to(BASE_DIR)), label))
    return rows


def main():
    tb_rows = collect_images(IMAGE_ROOT / "Tuberculosis", label=1)
    normal_rows = collect_images(IMAGE_ROOT / "Normal", label=0)

    all_rows = tb_rows + normal_rows

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_path", "label_tb"])
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {OUTPUT_CSV}")
    print(f"  Tuberculosis: {len(tb_rows)}")
    print(f"  Normal:       {len(normal_rows)}")
    print("Each image appears exactly once -- no synthetic duplication.")


if __name__ == "__main__":
    main()
