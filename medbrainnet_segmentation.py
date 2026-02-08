############################################
# ================ IMPORTS =================
############################################

import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy import ndimage
from skimage.morphology import remove_small_objects, closing, disk
from skimage.measure import label
from PIL import Image

############################################
# ============ DATA INSPECTION =============
############################################

DATA_DIR = "/kaggle/input/brats2020-training-data/BraTS2020_training_data/content/data"

h5_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".h5")]
print(f"Found {len(h5_files)} H5 files")

if len(h5_files) > 0:
    with h5py.File(h5_files[0], 'r') as f:
        print("Keys:", list(f.keys()))
        print("Image shape:", f['image'].shape)
        print("Mask shape:", f['mask'].shape)

############################################
# ============== AUGMENTATION ==============
############################################

class BrainScanAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < self.p:
            noise = torch.randn_like(image) * random.uniform(0.01, 0.05)
            image = torch.clamp(image + noise, 0, 1)

        return image, mask

############################################
# ================ DATASET ================
############################################

class BrainScanDataset(Dataset):
    def __init__(self, files, augment=False, patch_based=False, patch_size=128):
        self.files = files
        self.augment = augment
        self.patch_based = patch_based
        self.patch_size = patch_size
        self.aug = BrainScanAugmentation()

    def __len__(self):
        return len(self.files)

    def extract_brain_mask(self, image):
        thresh = np.percentile(image[0], 10)
        mask = image[0] > thresh
        mask = ndimage.binary_fill_holes(mask)
        return mask

    def standardize(self, image, brain_mask):
        out = np.zeros_like(image)
        for c in range(image.shape[0]):
            vals = image[c][brain_mask]
            if len(vals) > 0:
                mean, std = vals.mean(), vals.std()
                out[c] = (image[c] - mean) / (std + 1e-8)
        out = np.clip(out, -5, 5)
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        return out

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], 'r') as f:
            image = f['image'][()].transpose(2, 0, 1)
            mask = f['mask'][()].transpose(2, 0, 1)

        brain_mask = self.extract_brain_mask(image)
        image = self.standardize(image, brain_mask)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.augment:
            image, mask = self.aug(image, mask)

        return image, mask

############################################
# ========== TRAIN / VAL SPLIT ============
############################################

random.shuffle(h5_files)
split = int(0.8 * len(h5_files))
train_files = h5_files[:split]
val_files = h5_files[split:]

train_dataset = BrainScanDataset(train_files, augment=True)
val_dataset = BrainScanDataset(val_files)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

############################################
# ============== MODEL ====================
############################################

class ChannelAttention(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(c, c//r, 1),
            nn.ReLU(),
            nn.Conv2d(c//r, c, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(
            self.fc(F.adaptive_avg_pool2d(x, 1)) +
            self.fc(F.adaptive_max_pool2d(x, 1))
        )

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        max_, _ = torch.max(x, 1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, max_], 1)))

class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

def conv_block(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1),
        nn.BatchNorm2d(oc),
        nn.LeakyReLU(0.1),
        nn.Conv2d(oc, oc, 3, padding=1),
        nn.BatchNorm2d(oc),
        nn.LeakyReLU(0.1),
        CBAM(oc)
    )

class UNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = conv_block(4, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)

        self.pool = nn.MaxPool2d(2)
        self.center = conv_block(256, 512)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec1 = conv_block(32+64, 32)
        self.dec2 = conv_block(64+128, 64)
        self.dec3 = conv_block(128+256, 128)
        self.dec4 = conv_block(256+512, 256)

        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        c = self.center(self.pool(x4))

        d4 = self.dec4(torch.cat([x4, self.up(c)], 1))
        d3 = self.dec3(torch.cat([x3, self.up(d4)], 1))
        d2 = self.dec2(torch.cat([x2, self.up(d3)], 1))
        d1 = self.dec1(torch.cat([x1, self.up(d2)], 1))

        return self.out(d1)

############################################
# ================ LOSSES =================
############################################

class DiceLoss(nn.Module):
    def forward(self, x, y):
        x = torch.sigmoid(x).view(-1)
        y = y.view(-1)
        inter = (x * y).sum()
        return 1 - (2*inter + 1) / (x.sum() + y.sum() + 1)

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        return 0.6*self.dice(x,y) + 0.4*self.bce(x,y)

############################################
# ============== METRICS ==================
############################################

def dice_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    return (2*inter) / (pred.sum() + target.sum() + 1e-8)

############################################
# ============= TRAIN LOOP ================
############################################

def train(model, train_loader, val_loader, device, epochs=30):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler()
    crit = CombinedLoss()

    train_losses, val_dices = [], []
    best = 0

    for e in range(epochs):
        model.train()
        epoch_loss = 0
        for x,y in tqdm(train_loader, desc=f"Epoch {e+1}"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            with autocast():
                p = model(x)
                loss = crit(p,y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss/len(train_loader))
        d = validate(model, val_loader, device)
        val_dices.append(d)

        print(f"Epoch {e+1}: loss={train_losses[-1]:.4f}, dice={d:.4f}")

        if d > best:
            best = d
            torch.save({'model_state_dict': model.state_dict()}, "best_model.pth")

    return train_losses, val_dices

def validate(model, loader, device):
    model.eval()
    d = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            d += dice_score(model(x), y)
    return d / len(loader)

############################################
# ============ TRAIN PLOTS ================
############################################

def plot_history(losses, dices):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(losses)
    plt.title("Training Loss")

    plt.subplot(1,2,2)
    plt.plot(dices)
    plt.title("Validation Dice")
    plt.show()

############################################
# ============ ENSEMBLE ===================
############################################

class BrainSegmentationEnsemble(nn.Module):
    def __init__(self, paths, weights, device):
        super().__init__()
        self.models = []
        self.weights = weights
        for p in paths:
            m = UNetPlusPlus().to(device)
            m.load_state_dict(torch.load(p)['model_state_dict'])
            m.eval()
            self.models.append(m)

    def forward(self, x):
        preds = []
        for m in self.models:
            p0 = m(x)
            p1 = torch.flip(m(torch.flip(x,[3])),[3])
            p2 = torch.flip(m(torch.flip(x,[2])),[2])
            preds.append((p0+p1+p2)/3)
        out = sum(w*p for w,p in zip(self.weights,preds))
        return out / sum(self.weights)

############################################
# ============= VISUALIZATION =============
############################################

def visualize(image, pred):
    img = image[0,0].cpu()
    pred = (torch.sigmoid(pred)[0] > 0.5).cpu()

    overlay = np.zeros((*img.shape,3))
    overlay[pred[0]>0] = [0,0,1]
    overlay[pred[1]>0] = [0,1,0]
    overlay[pred[2]>0] = [1,0.6,0]

    plt.imshow(img, cmap='gray')
    plt.imshow(overlay, alpha=0.5)
    plt.axis('off')
    plt.show()

############################################
# ================= MAIN ==================
############################################

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlus().to(device)

    losses, dices = train(model, train_loader, val_loader, device)
    plot_history(losses, dices)

    ensemble = BrainSegmentationEnsemble(
        ["best_model.pth"],
        [1.0],
        device
    )

    x,y = next(iter(val_loader))
    x = x.to(device)

    with torch.no_grad():
        p = ensemble(x)

    visualize(x, p)

if __name__ == "__main__":
    main()
