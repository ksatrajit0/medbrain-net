import os
import numpy as np
from tqdm import tqdm
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
DATASET_DIR = "/kaggle/input/datasets/sartajbhuvaji/brain-tumor-classification-mri"
OUT_DIR = "/kaggle/working/KBTC_npy"
IMG_SIZE = (150, 150)
CLASS_NAMES = ["no_tumor", "pituitary_tumor", "meningioma_tumor", "glioma_tumor"]

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Helper function
# -------------------------
def load_images_and_labels(split_dir, class_names, img_size=(150, 150)):
    images = []
    labels = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} does not exist, skipping.")
            continue

        file_list = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for file_name in tqdm(file_list, desc=f"Loading {class_name}"):
            img_path = os.path.join(class_dir, file_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size, Image.BILINEAR)
                img_arr = np.array(img)
                images.append(img_arr)
                labels.append(class_idx)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")

    images = np.stack(images).astype(np.uint8)
    labels = np.array(labels).astype(np.int32)
    return images, labels

# -------------------------
# Main conversion
# -------------------------
def main():
    for split in ["Training", "Testing"]:
        split_dir = os.path.join(DATASET_DIR, split)
        images, labels = load_images_and_labels(split_dir, CLASS_NAMES, IMG_SIZE)

        images_path = os.path.join(OUT_DIR, f"KBTC_{split.lower()}_images.npy")
        labels_path = os.path.join(OUT_DIR, f"KBTC_{split.lower()}_labels.npy")

        np.save(images_path, images)
        np.save(labels_path, labels)

        print(f"{split} -> Saved {len(images)} images to {images_path}")
        print(f"{split} -> Saved labels to {labels_path}")
        print(f"{split} -> Images shape: {images.shape}, Labels shape: {labels.shape}")

if __name__ == "__main__":
    main()