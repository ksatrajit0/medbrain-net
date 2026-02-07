import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torch.utils.data import DataLoader
import h5py

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
CONFIG = {
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "n_filters": 32,
    "deep_supervision": True,
    "target_size": (240, 240),
    
    # Paths
    "checkpoint_paths": [
        '/kaggle/input/unetplusplus/pytorch/default/1/best_brain_segmentation_model.pth',
        '/kaggle/input/unetplusplus2/pytorch/default/1/best_brain_segmentation_model archi 2.pth'
    ],
    "ensemble_weights": [0.4, 0.6],
    "test_image_path": '/kaggle/input/picture/slice1_T1 (2).jpg',
    "output_dir": "./brain_segmentation_results",
    
    # Labels & Colors
    "tumor_classes": ['NCR', 'ED', 'ET'],
    "colors": [(0, 0, 1), (0, 1, 0), (1, 0.6, 0)] # Blue, Green, Orange
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==========================================
# 2. MODEL COMPONENTS (CBAM + UNet++)
# ==========================================
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        return x * self.sa(res)

def conv_block(in_c, out_c, dropout_p=0.2):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1, inplace=True),
        CBAM(out_c),
        nn.Dropout2d(dropout_p)
    )



class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, deep_supervision=True, n_filters=32):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.e1 = conv_block(in_channels, n_filters)
        self.e2 = conv_block(n_filters, n_filters*2)
        self.e3 = conv_block(n_filters*2, n_filters*4)
        self.e4 = conv_block(n_filters*4, n_filters*8)
        self.center = conv_block(n_filters*8, n_filters*16)

        self.d4_1 = conv_block(n_filters*8 + n_filters*16, n_filters*8)
        self.d3_2 = conv_block(n_filters*4 + n_filters*8, n_filters*4)
        self.d2_3 = conv_block(n_filters*2 + n_filters*4, n_filters*2)
        self.d1_4 = conv_block(n_filters + n_filters*2, n_filters)
        
        self.final = nn.Conv2d(n_filters, out_channels, 1)

    def forward(self, x):
        x1 = self.e1(x); x2 = self.e2(self.pool(x1)); x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3)); ct = self.center(self.pool(x4))
        
        x4_1 = self.d4_1(torch.cat([x4, self.up(ct)], 1))
        x3_2 = self.d3_2(torch.cat([x3, self.up(x4_1)], 1))
        x2_3 = self.d2_3(torch.cat([x2, self.up(x3_2)], 1))
        x1_4 = self.d1_4(torch.cat([x1, self.up(x2_3)], 1))
        
        return [self.final(x1_4)] if self.deep_supervision else self.final(x1_4)

# ==========================================
# 3. ENSEMBLE, METRICS & HISTORY
# ==========================================
class BrainSegmentationEnsemble(nn.Module):
    def __init__(self, paths, weights, device):
        super().__init__()
        self.models = nn.ModuleList()
        self.weights = weights
        for path in paths:
            m = UNetPlusPlus(n_filters=CONFIG["n_filters"], deep_supervision=CONFIG["deep_supervision"])
            ckpt = torch.load(path, map_location=device)
            m.load_state_dict(ckpt['model_state_dict'])
            m.to(device).eval()
            self.models.append(m)

    def forward(self, x):
        all_preds = []
        for model in self.models:
            p = model(x)[-1] if CONFIG["deep_supervision"] else model(x)
            # TTA logic (Horizontal & Vertical Flips)
            p_h = torch.flip(model(torch.flip(x, [3]))[-1 if CONFIG["deep_supervision"] else None], [3])
            p_v = torch.flip(model(torch.flip(x, [2]))[-1 if CONFIG["deep_supervision"] else None], [2])
            all_preds.append((p + p_h + p_v) / 3.0)
        return sum(w * pred for w, pred in zip(self.weights, all_preds)) / sum(self.weights)

def dice_coef(y_true, y_pred, epsilon=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = (torch.sigmoid(y_pred) > 0.5).float().flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + epsilon) / (y_true_f.sum() + y_pred_f.sum() + epsilon)

def plot_history(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    if 'train_loss' in history: plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history: plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for m in ['dice', 'iou', 'acc']:
        if f'val_{m}' in history: plt.plot(history[f'val_{m}'], label=f'Val {m.upper()}')
    plt.title('Metric History'); plt.legend(); plt.grid(True)
    plt.savefig(f"{CONFIG['output_dir']}/training_history.png")
    plt.show()

# ==========================================
# 4. MAIN EXECUTION (INFERENCE & VISUALS)
# ==========================================
def main():
    print(f"Using Device: {CONFIG['device']}")
    
    # 1. Load Ensemble
    ensemble = BrainSegmentationEnsemble(CONFIG["checkpoint_paths"], CONFIG["ensemble_weights"], CONFIG["device"])
    print("Ensemble Models Loaded.")

    # 2. Preprocess
    img = Image.open(CONFIG["test_image_path"]).convert('L').resize(CONFIG["target_size"])
    img_np = np.array(img).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(np.stack([img_np]*4)).unsqueeze(0).to(CONFIG["device"])

    # 3. Inference
    with torch.no_grad():
        output = ensemble(input_tensor)
        mask = (torch.sigmoid(output) > 0.5).cpu().numpy()[0]

    # 4. Visualization (MRI + 3 Classes + Combined)
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    axes[0].imshow(img_np, cmap='gray'); axes[0].axis('off'); axes[0].set_title("Original MRI")
    
    combined = np.zeros((*CONFIG["target_size"], 3))
    for i in range(3):
        overlay = np.zeros((*CONFIG["target_size"], 3))
        overlay[mask[i] > 0] = CONFIG["colors"][i]
        combined[mask[i] > 0] = CONFIG["colors"][i]
        axes[i+1].imshow(img_np, cmap='gray')
        axes[i+1].imshow(overlay, alpha=0.5)
        axes[i+1].set_title(CONFIG["tumor_classes"][i]); axes[i+1].axis('off')
        
    axes[4].imshow(img_np, cmap='gray')
    axes[4].imshow(combined, alpha=0.5)
    axes[4].set_title("Full Segmentation"); axes[4].axis('off')
    
    plt.savefig(f"{CONFIG['output_dir']}/final_prediction.png")
    plt.show()
    print(f"Results saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()