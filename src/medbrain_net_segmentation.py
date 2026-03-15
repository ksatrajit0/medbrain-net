import os
import h5py
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from segmentation_model import UNetPlusPlus
from segmentation_training import run_training
from segmentation_inference import MedBrainEnsemble, postprocess_and_plot

class BraTSDataset(Dataset):
    def __init__(self, list_IDs, data_path, transform=None):
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        file_path = os.path.join(self.data_path, ID)

        with h5py.File(file_path, 'r') as f:
            image = np.array(f.get('image'))  # (4,H,W)
            mask = f.get('mask')
            if mask is not None:
                mask = np.array(mask)  # (3,H,W)
            else:
                mask = None

        # Normalize each channel
        for c in range(image.shape[0]):
            img_c = image[c]
            if np.std(img_c) > 0:
                image[c] = (img_c - np.mean(img_c)) / np.std(img_c)

        # Brain mask (optional)
        brain_mask = (image[0] > image[0].min()).astype(np.float32)
        image = image * brain_mask

        if self.transform:
            img_hwc = image.transpose(1, 2, 0)
            if mask is not None:
                mask_hwc = mask.transpose(1, 2, 0)
            else:
                mask_hwc = None

            augmented = self.transform(image=img_hwc, mask=mask_hwc)
            image = augmented['image']
            mask = augmented['mask'] if mask_hwc is not None else None
        else:
            image = torch.from_numpy(image).float()
            if mask is not None:
                mask = torch.from_numpy(mask).float()

        return image, mask

def get_transforms(is_training=True):
    if is_training:
        return A.Compose([
            A.CenterCrop(height=160, width=160, p=1.0),
            A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(0.01, 0.05), p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.Sharpen(p=0.2),
                A.CoarseDropout(max_holes=8, max_height=12, max_width=12, p=0.3),
            ], p=0.4),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.CenterCrop(height=160, width=160, p=1.0),
            ToTensorV2()
        ])

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATA_DIR = './BraTS2020_H5'  # output from parser
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')

    train_files = sorted(os.listdir(TRAIN_DIR))
    val_files = sorted(os.listdir(VAL_DIR))
    test_files = sorted(os.listdir(TEST_DIR))

    train_loader = DataLoader(
        BraTSDataset(train_files, TRAIN_DIR, transform=get_transforms(True)),
        batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        BraTSDataset(val_files, VAL_DIR, transform=get_transforms(False)),
        batch_size=1, shuffle=False
    )
    test_loader = DataLoader(
        BraTSDataset(test_files, TEST_DIR, transform=get_transforms(False)),
        batch_size=1, shuffle=False
    )

    print(f"--- MedBrain-Net: Clinical Segmentation Pathway Initiated ---")

    model_instance = UNetPlusPlus(in_channels=4, out_channels=3, deep_supervision=True).to(DEVICE)
    run_training(model_instance, train_loader, val_loader, DEVICE)

    print("\n[INFERENCE] Executing Ensemble + TTA Pipeline...")
    checkpoint_paths = [
        "medbrain_run1.pth",
        "medbrain_run2.pth",
        "medbrain_run3.pth"
    ]
    weights = [0.33, 0.33, 0.34]

    ensemble = MedBrainEnsemble(checkpoint_paths, DEVICE, weights=weights)

    # Example inference on one test sample
    sample_img, _ = next(iter(test_loader))
    sample_img = sample_img.to(DEVICE)
    prediction = ensemble(sample_img)
    postprocess_and_plot(sample_img, prediction)

    print("\nWorkflow Complete. Results stored.")

if __name__ == "__main__":
    main()