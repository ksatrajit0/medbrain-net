import torch
import numpy as np
import matplotlib.pyplot as plt
from segmentation_model import UNetPlusPlus
from skimage.morphology import remove_small_objects, closing, disk

class MedBrainEnsemble(torch.nn.Module):
    def __init__(self, paths, device, weights=[0.4, 0.6]):
        super().__init__()
        self.models = torch.nn.ModuleList()
        self.weights = weights
        self.device = device
        
        for p in paths:
            
            m = UNetPlusPlus(in_channels=4, out_channels=3, deep_supervision=True)
            checkpoint = torch.load(p, map_location=device)
            m.load_state_dict(checkpoint['model_state_dict'])
            m.to(device).eval()
            self.models.append(m)

    def forward(self, x):

        final_pred = 0
        with torch.no_grad():
            for i, model in enumerate(self.models):

                p0 = model(x)[-1] 
                
                p1_raw = model(torch.flip(x, [3]))[-1]
                p1 = torch.flip(p1_raw, [3])

                p2_raw = model(torch.flip(x, [2]))[-1]
                p2 = torch.flip(p2_raw, [2])
                
                tta_avg = (p0 + p1 + p2) / 3.0
                final_pred += self.weights[i] * tta_avg
                
        return final_pred

def postprocess_and_plot(image_tensor, pred_tensor):

    probs = torch.sigmoid(pred_tensor)
    mask = (probs > 0.5).cpu().numpy().astype(np.uint8)[0] # Shape: (3, H, W)
    
    for c in range(3):
        
        mask[c] = remove_small_objects(mask[c].astype(bool), min_size=64)
        
        mask[c] = closing(mask[c], disk(2))
    
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    background = image_tensor[0, 0].cpu().numpy()
    
    ax[0].imshow(background, cmap='gray')
    ax[0].set_title("Input MRI (Standardized)")
    ax[0].axis('off')
    
    overlay = np.zeros((*mask[0].shape, 3))
    overlay[..., 0] = mask[0]
    overlay[..., 1] = mask[1]
    overlay[..., 2] = mask[2]
    
    ax[1].imshow(background, cmap='gray')
    ax[1].imshow(overlay, alpha=0.4)
    ax[1].set_title("MedBrain-Net Ensemble Prediction")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()