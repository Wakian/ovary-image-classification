import os
import cv2
import shutil
import random
import numpy as np
import albumentations as A
from pathlib import Path

# ============================================================
# SETTINGS - CHANGE THESE TO MATCH YOUR SETUP
# ============================================================
DATASET_PATH = "C:\\Users\\brian\\Desktop\\Uni\\sistemas_int\\ovary\\data_sets\\ovarian_ultrasound_dataset"
OUTPUT_PATH  = "C:\\Users\\brian\\Desktop\\Uni\\sistemas_int\\ovary\\data_sets\\augmented_1000_ovarian_ultrasound_dataset"

TARGET_TRAIN_IMAGES_PER_CLASS = 1000   # how many images train will have per class after augmentation
TEST_SPLIT_RATIO              = 0.2    # fraction of ORIGINAL images reserved for test (0.2 = 20%)
RANDOM_SEED                   = 42     # fixed seed so the split is identical every run

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
# ============================================================
# OUTPUT STRUCTURE:
#
#   OUTPUT_PATH/
#     train/
#       dominant_follicle/   <- original train images + augmented images
#       normal/
#       PCO/
#     test/
#       dominant_follicle/   <- ONLY original images, never augmented
#       normal/
#       PCO/
# ============================================================


def get_augmentation_pipeline():
    """Safe augmentations for medical ultrasound images."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.6),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1,
            rotate_limit=10, border_mode=cv2.BORDER_REFLECT, p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.ElasticTransform(alpha=30, sigma=5, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(10, 30),
            hole_width_range=(10, 30),
            fill=0, p=0.2
        ),
    ])


def load_images_from_folder(folder_path):
    """Load all unique image paths from a folder."""
    seen = set()
    unique_images = []
    for path in Path(folder_path).iterdir():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_images.append(path)
    return unique_images


def split_images(image_paths, test_ratio, seed):
    """
    Randomly split image paths into (train, test) on ORIGINAL images only.
    Fixed seed ensures the split is identical every time you run the script.
    This must happen BEFORE augmentation so no augmented version of a test
    image can ever appear in the training set.
    """
    random.seed(seed)
    shuffled = image_paths.copy()
    random.shuffle(shuffled)

    n_test  = max(1, int(len(shuffled) * test_ratio))
    train_paths = shuffled[:-n_test]
    test_paths  = shuffled[-n_test:]
    return train_paths, test_paths


def copy_images(image_paths, output_folder):
    """Copy original images to a destination folder."""
    os.makedirs(output_folder, exist_ok=True)
    copied = 0
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: Could not read {img_path.name}, skipping")
            continue
        cv2.imwrite(os.path.join(output_folder, img_path.name), img)
        copied += 1
    return copied


def augment_train_folder(train_image_paths, output_folder, target_count, transform):
    """
    Augment only the train folder up to target_count.
    train_image_paths are the source images (originals only).
    The test folder is never passed here and is never touched.
    """
    current_count = len(train_image_paths)
    needed = target_count - current_count

    if needed <= 0:
        print(f"  Train already has {current_count} images, no augmentation needed.")
        return

    aug_saved = 0
    attempts  = 0
    max_attempts = needed * 5

    while aug_saved < needed and attempts < max_attempts:
        img_path = train_image_paths[attempts % current_count]
        img = cv2.imread(str(img_path))
        attempts += 1

        if img is None:
            print(f"  WARNING: Could not read {img_path.name}, skipping")
            continue

        aug_img  = transform(image=img)["image"]
        out_name = f"aug_{aug_saved:05d}_{img_path.stem}.jpg"
        out_path = os.path.join(output_folder, out_name)

        if cv2.imwrite(out_path, aug_img):
            aug_saved += 1
        else:
            print(f"  WARNING: Failed to write {out_name}")

    if aug_saved < needed:
        print(f"  WARNING: Only saved {aug_saved}/{needed} augmented images")
    else:
        print(f"  Generated {aug_saved} augmented images")


def print_split_summary(output_path):
    """Print train vs test counts per class."""
    print("\n  Final Dataset Summary:")
    print(f"  {'Class':<22} {'Train':>7} {'Test':>6} {'Total':>7}")
    print("  " + "-" * 44)

    train_root = Path(output_path) / "train"
    test_root  = Path(output_path) / "test"

    all_classes = sorted(set(
        [f.name for f in train_root.iterdir() if f.is_dir()] +
        [f.name for f in test_root.iterdir()  if f.is_dir()]
    ))

    for cls in all_classes:
        n_train = len(load_images_from_folder(train_root / cls)) if (train_root / cls).exists() else 0
        n_test  = len(load_images_from_folder(test_root  / cls)) if (test_root  / cls).exists() else 0
        print(f"  {cls:<22} {n_train:>7} {n_test:>6} {n_train + n_test:>7}")

    print("  " + "-" * 44)


def main():
    print("=" * 54)
    print("  Ovarian Ultrasound Image Augmentation v3")
    print("  (train/test split BEFORE augmentation)")
    print("=" * 54)

    if not os.path.exists(DATASET_PATH):
        print(f"\nERROR: Dataset folder not found: {DATASET_PATH}")
        return

    class_folders = [f for f in Path(DATASET_PATH).iterdir() if f.is_dir()]
    if not class_folders:
        print("ERROR: No class subfolders found in the dataset folder.")
        return

    transform = get_augmentation_pipeline()

    print(f"\nSettings:")
    print(f"  Test split          : {int(TEST_SPLIT_RATIO * 100)}% of original images")
    print(f"  Train target        : {TARGET_TRAIN_IMAGES_PER_CLASS} images per class")
    print(f"  Random seed         : {RANDOM_SEED}")
    print(f"  Output              : {OUTPUT_PATH}\n")

    for class_folder in sorted(class_folders):
        cls_name = class_folder.name
        print(f"\nProcessing class: {cls_name}")
        print("  " + "-" * 40)

        # STEP 1 — load all original images
        all_images = load_images_from_folder(class_folder)
        print(f"  Original images found : {len(all_images)}")

        # STEP 2 — split originals into train/test BEFORE any augmentation
        train_imgs, test_imgs = split_images(all_images, TEST_SPLIT_RATIO, RANDOM_SEED)
        print(f"  Train split           : {len(train_imgs)} images")
        print(f"  Test split            : {len(test_imgs)} images")

        # STEP 3 — copy each split to its own output folder
        train_out = os.path.join(OUTPUT_PATH, "train", cls_name)
        test_out  = os.path.join(OUTPUT_PATH, "test",  cls_name)

        copied_train = copy_images(train_imgs, train_out)
        copied_test  = copy_images(test_imgs,  test_out)
        print(f"  Copied to train/      : {copied_train}")
        print(f"  Copied to test/       : {copied_test}  <- will NOT be augmented")

        # STEP 4 — augment ONLY the train folder
        # test_out is never passed here, guaranteeing zero leakage
        print(f"  Augmenting train/ to {TARGET_TRAIN_IMAGES_PER_CLASS} total...")
        augment_train_folder(train_imgs, train_out, TARGET_TRAIN_IMAGES_PER_CLASS, transform)

        actual_train = len(load_images_from_folder(train_out))
        actual_test  = len(load_images_from_folder(test_out))
        print(f"  Final train/ count    : {actual_train}")
        print(f"  Final test/ count     : {actual_test}  (originals only)")

    print("\n" + "=" * 54)
    print("  Augmentation Complete!")
    print_split_summary(OUTPUT_PATH)
    print(f"\nDataset saved to: {OUTPUT_PATH}")
    print("  -> Load train/ into Orange for training")
    print("  -> Use  test/  for final evaluation only\n")


if __name__ == "__main__":
    main()