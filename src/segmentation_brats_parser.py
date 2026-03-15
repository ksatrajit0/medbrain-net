import os
import nibabel as nib
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# -----------------------------
# Dataset Paths
# -----------------------------
DATA_ROOT = "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation"

TRAIN_ROOT = os.path.join(
    DATA_ROOT, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData"
)

VAL_ROOT = os.path.join(
    DATA_ROOT, "BraTS2020_ValidationData", "MICCAI_BraTS2020_ValidationData"
)

OUTPUT_ROOT = "/kaggle/working/BraTS2020_H5"

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT,"train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT,"val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT,"test"), exist_ok=True)

# -----------------------------
# Normalization function
# -----------------------------
def normalize(img):
    mask = img > 0
    if np.sum(mask) == 0:
        return img
    mean = img[mask].mean()
    std = img[mask].std()
    return (img - mean) / (std + 1e-8)

# -----------------------------
# Convert seg.nii to 3-channel mask
# -----------------------------
def convert_mask(mask_slice):
    mask = np.stack([
        (mask_slice == 1),
        (mask_slice == 2),
        (mask_slice == 4)
    ], axis=0).astype(np.float32)
    return mask

# -----------------------------
# Save slice to H5
# -----------------------------
def save_slice(image, mask, split, name):
    path = os.path.join(OUTPUT_ROOT, split, name)
    with h5py.File(path, "w") as f:
        f.create_dataset("image", data=image)
        if mask is not None:
            f.create_dataset("mask", data=mask)

# -----------------------------
# TRAIN / VAL PARSER
# -----------------------------
print("Scanning training patients...")

patients = sorted([
    p for p in os.listdir(TRAIN_ROOT)
    if p.startswith("BraTS20_Training")
])

print("Total training patients:", len(patients))

train_ids, val_ids = train_test_split(
    patients, test_size=0.2, random_state=42
)

splits = {"train": train_ids, "val": val_ids}

for split_name, plist in splits.items():
    print(f"\nProcessing {split_name} set")
    for patient in tqdm(plist):
        pdir = os.path.join(TRAIN_ROOT, patient)

        flair = nib.load(os.path.join(pdir,f"{patient}_flair.nii")).get_fdata()
        t1 = nib.load(os.path.join(pdir,f"{patient}_t1.nii")).get_fdata()
        t1ce = nib.load(os.path.join(pdir,f"{patient}_t1ce.nii")).get_fdata()
        t2 = nib.load(os.path.join(pdir,f"{patient}_t2.nii")).get_fdata()
        seg = nib.load(os.path.join(pdir,f"{patient}_seg.nii")).get_fdata()

        # normalize each modality
        flair = normalize(flair)
        t1 = normalize(t1)
        t1ce = normalize(t1ce)
        t2 = normalize(t2)

        image = np.stack([flair,t1,t1ce,t2], axis=0)
        depth = image.shape[-1]

        for slice_idx in range(depth):
            img_slice = image[:,:,:,slice_idx]
            mask_slice = seg[:,:,slice_idx]
            mask = convert_mask(mask_slice)

            # skip empty slices
            if np.sum(mask) == 0:
                continue

            name = f"{patient}_{slice_idx}.h5"
            save_slice(img_slice, mask, split_name, name)

# -----------------------------
# TEST PARSER
# -----------------------------
print("\nProcessing TEST set")

test_patients = sorted([
    p for p in os.listdir(VAL_ROOT)
    if p.startswith("BraTS20_Validation")
])

print("Total test patients:", len(test_patients))

for patient in tqdm(test_patients):
    pdir = os.path.join(VAL_ROOT, patient)

    flair = nib.load(os.path.join(pdir,f"{patient}_flair.nii")).get_fdata()
    t1 = nib.load(os.path.join(pdir,f"{patient}_t1.nii")).get_fdata()
    t1ce = nib.load(os.path.join(pdir,f"{patient}_t1ce.nii")).get_fdata()
    t2 = nib.load(os.path.join(pdir,f"{patient}_t2.nii")).get_fdata()

    # normalize
    flair = normalize(flair)
    t1 = normalize(t1)
    t1ce = normalize(t1ce)
    t2 = normalize(t2)

    image = np.stack([flair,t1,t1ce,t2], axis=0)
    depth = image.shape[-1]

    for slice_idx in range(depth):
        img_slice = image[:,:,:,slice_idx]
        name = f"{patient}_{slice_idx}.h5"
        save_slice(img_slice, None, "test", name)

print("\nPreprocessing Complete!")
