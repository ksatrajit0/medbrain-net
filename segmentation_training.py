import torch
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler, autocast

class HybridLoss(torch.nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1e-6):
        inputs_sig = torch.sigmoid(inputs)
        bce = torch.nn.functional.binary_cross_entropy(inputs_sig, targets, reduction='mean')
        
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)
        
        bce_log = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_log)
        focal_loss = (self.alpha * (1 - pt)**self.gamma * bce_log).mean()

        return bce + dice + focal_loss

def calculate_metrics(pred, target, threshold=0.5):

    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + 1e-7) / (union + 1e-7)
    iou = (intersection + 1e-7) / (union - intersection + 1e-7)
    return dice.item(), iou.item()

def run_training(model, train_loader, val_loader, device):
    criterion = HybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, steps_per_epoch=len(train_loader), epochs=30
    )
    scaler = GradScaler()

    best_dice = 0.0
    patience = 7
    counter = 0

    for epoch in range(30):

        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = sum(criterion(out, masks) for out in outputs) / 4
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
        
        model.eval()
        val_dice, val_iou = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                # Use the final high-res output for metrics
                d, i = calculate_metrics(outputs[-1], masks)
                val_dice.append(d)
                val_iou.append(i)

        avg_dice = np.mean(val_dice)
        avg_iou = np.mean(val_iou)
        
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            counter = 0
            torch.save({'model_state_dict': model.state_dict()}, "medbrain_final.pth")
            print(f"--> Strategy: Best Model Saved (Dice: {best_dice:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print("Early Stopping Triggered.")
                break