import random
import shutil
from pathlib import Path

import pandas as pd


def prepare_data_and_split(
    base_path: Path = Path("data/emergency_vehicles/Emergency_Vehicles"),
    output_dir: Path = Path("data/emergency_yolo_cls"),
    val_ratio: float = 0.2,
    seed: int = 42,
):
    train_csv = base_path / "train.csv"
    train_images = base_path / "train"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train.csv at: {train_csv.resolve()}")
    if not train_images.exists():
        raise FileNotFoundError(f"Missing train images dir at: {train_images.resolve()}")

    random.seed(seed)

    # 1) Build train folders by copying images
    df = pd.read_csv(train_csv)

    train_root = output_dir / "train"
    val_root = output_dir / "val"

    # Optional: start clean (uncomment if you want fresh rebuild each time)
    # if output_dir.exists():
    #     shutil.rmtree(output_dir)

    copied = 0
    skipped = 0

    for _, row in df.iterrows():
        img_name = str(row["image_names"])
        is_emergency = int(row["emergency_or_not"])  # 1=emergency, 0=non_emergency

        label = "emergency" if is_emergency == 1 else "non_emergency"
        target_folder = train_root / label
        target_folder.mkdir(parents=True, exist_ok=True)

        src = train_images / img_name
        dst = target_folder / img_name

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            skipped += 1

    print(f"[1/2] Copied {copied} images into: {train_root.resolve()} (skipped: {skipped})")

    # 2) Create val split by moving a fraction from train -> val
    for class_name in ["emergency", "non_emergency"]:
        src_class = train_root / class_name
        dst_class = val_root / class_name
        dst_class.mkdir(parents=True, exist_ok=True)

        images = list(src_class.glob("*.jpg"))
        if not images:
            print(f"[WARN] No images found for class '{class_name}' in {src_class}")
            continue

        n_val = int(len(images) * val_ratio)
        n_val = max(1, n_val) if len(images) > 1 else 0  # avoid moving the only image
        if n_val == 0:
            print(f"[WARN] Too few images to create val split for '{class_name}' (count={len(images)})")
            continue

        val_samples = random.sample(images, n_val)
        for img in val_samples:
            shutil.move(str(img), str(dst_class / img.name))

        print(f"[2/2] Moved {n_val}/{len(images)} '{class_name}' images to val.")

    # Summary counts
    def count_imgs(p: Path) -> int:
        return len(list(p.glob("**/*.jpg")))

    total_train = count_imgs(train_root)
    total_val = count_imgs(val_root)

    print("\nDone ✅")
    print(f"Train images: {total_train}")
    print(f"Val images:   {total_val}")
    print(f"Output dir:   {output_dir.resolve()}")


if __name__ == "__main__":
    prepare_data_and_split()
