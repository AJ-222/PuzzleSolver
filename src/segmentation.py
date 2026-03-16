import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.tta as tta
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.losses import DiceLoss

class CombinedLoss(nn.Module):
    """
    Combines BCEWithLogitsLoss and DiceLoss.
    alpha: weight for BCE, (1-alpha) for Dice
    """
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(mode='binary')
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

class PuzzleSegmentationDataset(Dataset):
    """Dataset for loading puzzle images and their corresponding binary masks."""
    def __init__(self, image_dir, mask_dir, image_paths, transform=None, is_test=False):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = image_paths
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.is_test:
            img_h, img_w = img.shape[:2]
            if self.transform:
                augmented = self.transform(image=img)
                img = augmented['image']
            return img, os.path.basename(img_path), (img_h, img_w)

        filename = os.path.basename(img_path)
        filename_stem = os.path.splitext(filename)[0]
        mask_path = os.path.join(self.mask_dir, f"{filename_stem}_mask.png")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask for path: {mask_path}")
            
        mask = (mask > 127).astype("float32")
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        mask = mask.unsqueeze(0)
        return img, mask

def get_segmentation_model(device, encoder_name="efficientnet-b4", weights="imagenet"):
    """Initializes the U-Net model with the specified encoder."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=weights,
        in_channels=3,
        classes=1,
    ).to(device)
    return model

def get_segmentation_transforms(is_train, img_size=(512, 512)):
    """Returns Albumentations transforms for training or inference."""
    if is_train:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def generate_masks_with_tta(model, dataloader, device, output_dir):
    """Runs inference using Test Time Augmentation and saves predicted masks."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    tta_model = tta.TTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
    tta_model.eval() 
    
    with torch.no_grad():
        for images, filenames, original_sizes in dataloader:
            images = images.to(device)
            predictions = tta_model(images)
            preds_binary = (torch.sigmoid(predictions) > 0.5).float()
            
            for i in range(preds_binary.shape[0]):
                mask_tensor = preds_binary[i].cpu()
                orig_h, orig_w = original_sizes[0][i].item(), original_sizes[1][i].item()

                mask_resized = F.interpolate(
                    mask_tensor.unsqueeze(0), 
                    size=(orig_h, orig_w), 
                    mode='nearest'
                ).squeeze() 
                
                mask_to_save = (mask_resized.numpy() * 255).astype(np.uint8) 
                filename_stem = os.path.splitext(filenames[i])[0]
                output_path = os.path.join(output_dir, f"{filename_stem}_mask.png")
                cv2.imwrite(output_path, mask_to_save)