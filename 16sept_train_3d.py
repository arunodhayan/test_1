
# ==================================
# === 1. Standard Library Imports ====
# ==================================
import os
import gc
import sys
import json
import shutil
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# ==================================
# === 2. Third-Party Imports =======
# ==================================
# --- Data Handling ---
import numpy as np
import pandas as pd
import polars as pl  # Note: You have both pandas and polars

# --- Image & Medical Data ---
import cv2
import pydicom
import timm
import timm_3d
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandFlipd,
    RandAffined,
    RandAdjustContrastd,
    RandGaussianNoised,
    ToTensord,ScaleIntensityRanged
)

# --- Machine Learning (PyTorch & Scikit-learn) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchmetrics.classification import AUROC
from transformers import get_cosine_schedule_with_warmup

# --- Utilities ---
from tqdm import tqdm
from IPython.display import display

# ==================================
# === 3. Your Local Imports (if any) ======
# ==================================
# from your_local_module import ...


# --- Initial Setup ---
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility in random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU setups
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Ensure that operations are deterministic on CuDNN (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CFG:
    # --- General ---
    comments = "EfficientNetV2-S_updated"
    fold = 0
    seed = 42

    # --- Paths ---
    files = "16newsept_preprocessed_data_128_rescale_corrected/train_preprocessed.csv"

    # --- Dataloader ---
    image_size = 128
    num_slices_sample = 128
    num_workers = 10
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 2
    n_splits = 5   
    # --- Model ---
    
    model_name =  "tf_efficientnetv2_s.in21k_ft_in1k"
    drop_rate= 0.3
    drop_path_rate= 0.2
    in_chans= 1
    n_slice_per_c= 128
    meta_dim = 0
    num_classes= 14
    
    # --- Training ---
    epochs = 20
    batch_size = 16
    use_amp = True
    early_stopping_patience = 10

    # --- Optimizer & Scheduler ---
    lr_t_max = 2e-4
    weight_decay = 1e-2
    lr_eta_min = 1e-6
    warmup_epochs = 2

    # --- Loss Function ---
    label_cols = [
        "Left Infraclinoid Internal Carotid Artery","Right Infraclinoid Internal Carotid Artery",
        "Left Supraclinoid Internal Carotid Artery","Right Supraclinoid Internal Carotid Artery",
        "Left Middle Cerebral Artery","Right Middle Cerebral Artery",
        "Anterior Communicating Artery","Left Anterior Cerebral Artery",
        "Right Anterior Cerebral Artery","Left Posterior Communicating Artery",
        "Right Posterior Communicating Artery","Basilar Tip",
        "Other Posterior Circulation","Aneurysm Present"
    ]
    loss_weights = [1.0]*13 + [13.0]

    # --- [CORRECTED UPDATE] Using YOUR Calculated Normalization Stats ---
    # These values are taken directly from the output you provided.
    STATS = {
        'CT':       {'a_min': 0.12,   'a_max': 245.46},
        'MRI_T1':   {'a_min': 0.00,   'a_max': 220.77},
        'MRI_T2':   {'a_min': 0.00,   'a_max': 228.84},
        # Using the overall MRI stats as a fallback for other MR types (like MRA)
        'MRI_Other':{'a_min': 0.01,   'a_max': 221.77} 
    }
NUM_TOTAL_CLASSES = len(CFG.label_cols)

# ==========================================================
# === 2. ANEURYSM-AWARE SAMPLING FUNCTION ==================
# ==========================================================
def sample_or_pad_volume(volume: np.ndarray, target_depth: int, annotations: list = None) -> np.ndarray:
    orig_depth, H, W = volume.shape

    if orig_depth < target_depth:
        pad_amount = target_depth - orig_depth
        last_slice = volume[-1:, :, :]
        padding = np.repeat(last_slice, pad_amount, axis=0)
        return np.concatenate([volume, padding], axis=0)

    if orig_depth > target_depth:
        start_slice = 0
        is_positive = annotations and len(annotations) > 0
        
        if is_positive:
            ann_slice_idx = annotations[0].get('slice_index', orig_depth // 2)
            min_start = max(0, ann_slice_idx - target_depth + 1)
            max_start = min(orig_depth - target_depth, ann_slice_idx)
            if min_start > max_start: min_start = max_start
            start_slice = np.random.randint(min_start, max_start + 1)
        else:
            start_slice = np.random.randint(0, orig_depth - target_depth + 1)
            
        end_slice = start_slice + target_depth
        return volume[start_slice:end_slice, :, :]

    return volume

# ==========================================================
# === 3. FINAL PYTORCH DATASET CLASS =======================
# ==========================================================
class RSNADataset3DFinal(Dataset):
    def __init__(self, df, transform=None):
        """
        df: DataFrame with volume_path, metadata_path, labels
        transform: optional MONAI transforms
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_cols = CFG.label_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- load volume + metadata
        vol = np.load(row["volume_path"]).astype(np.float32)  #/ 255.0
        with open(row["metadata_path"], 'r') as f:
            metadata = json.load(f)
        
        # --- sample/pad to fixed depth
        annotations = metadata.get('annotations', [])
#        print (annotations)
        vol = sample_or_pad_volume(vol, CFG.num_slices_sample, annotations=annotations)

        # --- wrap in MONAI dict
        #data = {"image": vol}

        # --- get modality
        modality = metadata.get("modality", "CT")
        #print (modality)

        # --- choose normalization strategy
        if modality == "CT":
            # fixed HU window
            a_min, a_max = 0.12, 600
        elif modality == "MR":
            # per-volume percentiles (robust to scanner differences)
            a_min, a_max = 0.0 , 221.77


        data = {"image": vol} 
        # --- normalization
        normalizer = ScaleIntensityRanged(
            keys=["image"],
            a_min=float(a_min), a_max=float(a_max),
            b_min=0.0, b_max=1.0,
            clip=True
        )
        data = normalizer(data)

        # --- apply optional augmentations
        if self.transform is not None:
            data = self.transform(data)

        # --- prepare outputs
        img = data["image"]
        meta = torch.tensor([
            metadata.get("pixel_spacing_x", 0.5),
            metadata.get("pixel_spacing_y", 0.5),
            metadata.get("z_spacing", 1.0)
        ], dtype=torch.float32)
        labels = torch.tensor(
            row[self.label_cols].values.astype(np.float32),
            dtype=torch.float32
        )
        
        return img, meta, labels


# ==========================================================
# === 4. DEFINE TRANSFORMS =================================
# ==========================================================
train_transforms = Compose([
    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0, 1, 2]),
    RandAffined(
        keys=['image'],
        prob=0.8,
        rotate_range=(np.pi/24, np.pi/24, np.pi/24),
        scale_range=(0.9, 1.1),
        mode=('bilinear'),
        padding_mode='border',
    ),
    ToTensord(keys=["image"]),
])

val_transforms = Compose([
    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    ToTensord(keys=["image"]),
])










# === MODEL ===


class Pure3DModel(nn.Module):
    """
    A pure 3D CNN model that preserves spatial information until the final layer.
    Ideal for localization tasks.
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int = 14, # Total classes (1 presence + 13 locations)
        pretrained: bool = True,
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.2,
        in_chans: int = 1,
        meta_dim: int = 0
    ):
        super().__init__()
        self.meta_dim = int(meta_dim)

        # Create the 3D backbone, but this time WITH a global pooling layer.
        # The backbone will output one feature vector per 3D volume.
        self.backbone = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,          # We still want our own classifier
            global_pool='avg',    # Use 'avg', 'max', or 'gem'
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )
        num_backbone_features = self.backbone.num_features

        # Optional metadata tower (same as before)
        if self.meta_dim > 0:
            self.meta_fc = nn.Sequential(
                nn.Linear(self.meta_dim, 16), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(16, 32), nn.ReLU()
            )
            final_in = num_backbone_features + 32
        else:
            self.meta_fc = None
            final_in = num_backbone_features

        # Classifier
        self.classifier = nn.Linear(final_in, num_classes)

    def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        # The forward pass is much simpler now
        feats = self.backbone(x) # Shape: (B, C)

        if self.meta_fc is not None and meta is not None and self.meta_dim > 0:
            m = self.meta_fc(meta)
            feats = torch.cat([feats, m], dim=1)

        return self.classifier(feats)






############################################################################################





# -----------------------
# One epoch: train (no meta)
# -----------------------
# ==================================
# === 7. EPOCH & FIT FUNCTIONS =====
# ==================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, meta, labels in tqdm(loader, desc="Training"):
        images, meta, labels = images.to(device), meta.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(enabled=CFG.use_amp):
            outputs = model(images, meta=meta)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        all_preds.append(torch.sigmoid(outputs).detach().cpu())
        all_labels.append(labels.detach().cpu())
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, torch.cat(all_preds), torch.cat(all_labels)

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, meta, labels in tqdm(loader, desc="Validation"):
        images, meta, labels = images.to(device), meta.to(device), labels.to(device)
        with autocast(enabled=CFG.use_amp):
            outputs = model(images, meta=meta)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        all_preds.append(torch.sigmoid(outputs).detach().cpu())
        all_labels.append(labels.detach().cpu())
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, torch.cat(all_preds), torch.cat(all_labels)

def fit_one_fold(df, train_tfms, val_tfms, fold_id):
    train_df = df[df['fold'] != fold_id].reset_index(drop=True)
    val_df = df[df['fold'] == fold_id].reset_index(drop=True)
    
    train_ds = RSNADataset3DFinal(df=train_df, transform=train_tfms)
    val_ds = RSNADataset3DFinal(df=val_df, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size * 2, shuffle=False, num_workers=CFG.num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pure3DModel(
        model_name=CFG.model_name, num_classes=CFG.num_classes, pretrained=True,
        drop_rate=CFG.drop_rate, drop_path_rate=CFG.drop_path_rate,
        in_chans=1, meta_dim=CFG.meta_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr_t_max, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.lr_eta_min)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(CFG.loss_weights).to(device))
    #criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(CFG.loss_weights).to(device))
    scaler = GradScaler(enabled=CFG.use_amp)
    metric_auroc = AUROC(task="multilabel", num_labels=CFG.num_classes, average=None).to(device)

    best_score, patience_counter, best_AP,best_LOC = -1.0, 0,-1.0,-1.0
    best_val_preds, best_val_labels = None, None
    aneurysm_idx = CFG.label_cols.index("Aneurysm Present")

    for epoch in range(CFG.epochs):
        print(f"\nEpoch {epoch+1}/{CFG.epochs}")
        tr_loss, _, _ = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_preds, val_labels = validate_one_epoch(model, val_loader, criterion, device)
        
        val_auc_per_class = metric_auroc(val_preds.to(device), val_labels.long().to(device))
        val_auc_ap = val_auc_per_class[aneurysm_idx].item()
        val_auc_loc = torch.mean(val_auc_per_class[[i for i in range(CFG.num_classes) if i != aneurysm_idx]]).item()
        val_score = 0.5 * (val_auc_ap + val_auc_loc)
        
        print(f"Val Loss: {val_loss:.4f} | Score: {val_score:.4f} (AP: {val_auc_ap:.4f}, LOC: {val_auc_loc:.4f})")
        scheduler.step()

        if val_score > best_score:
            print(f"Score improved ({best_score:.4f} --> {val_score:.4f}). Saving model.")
            best_score = val_score
            best_AP = val_auc_ap
            best_LOC = val_auc_loc
            patience_counter = 0
            torch.save(model.state_dict(), f"{CFG.comments}_model_fold_{fold_id}.pth")
            best_val_preds = val_preds
            best_val_labels = val_labels
        else:
            patience_counter += 1
            if patience_counter >= CFG.early_stopping_patience:
                print("Early stopping triggered.")
                break
    
    print(f"\nBest score for fold {fold_id}: {best_score:.4f}")
    return best_score, best_AP, best_LOC 


    

def run_kfold_training(df: pd.DataFrame, train_tfms, val_tfms, n_splits: int):
    set_seed(CFG.seed)
    y = df["Aneurysm Present"].astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG.seed)
    
    oof_preds = np.zeros((len(df), CFG.num_classes), dtype=np.float32)
    oof_labels = np.zeros((len(df), CFG.num_classes), dtype=np.float32)
    fold_metrics = []
    score =[]
    AP_score = []
    LOC_score = [] 
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(df, y)):
        print(f"\n========== Fold {fold_id} / {n_splits-1} ==========")
        df_fold = df.copy()
        df_fold['fold'] = -1
        df_fold.loc[va_idx, 'fold'] = fold_id
        
        best_score, best_AP, best_LOC = fit_one_fold(df_fold, train_tfms, val_tfms, fold_id)
        score.append(best_score)
        AP_score.append(best_AP)
        LOC_score.append(best_LOC)
                 


        
        print(f"[Fold {fold_id}] Best: AUC_AP={best_AP:.4f} | "
              f"AUC_LOC={best_LOC:.4f} | Score={best_score:.4f}")

        #del model
        gc.collect()
        torch.cuda.empty_cache()

    # --- Overall OOF metrics ---
    
    avg_score = np.mean(score)
    avg_AP = np.mean(AP_score)
    avg_LOC = np.mean(LOC_score)

    # Best single fold
    best_fold_idx = int(np.argmax(score))
    best_fold_score = score[best_fold_idx]
    best_fold_AP = AP_score[best_fold_idx]
    best_fold_LOC = LOC_score[best_fold_idx]

    print("===== Cross-validation summary =====")
    print(f"Avg Score : {avg_score:.4f}")
    print(f"Avg AP    : {avg_AP:.4f}")
    print(f"Avg LOC   : {avg_LOC:.4f}")
    print("-----------------------------------")
    print(f"Best Fold : {best_fold_idx}")
    print(f"Best Score: {best_fold_score:.4f}")
    print(f"Best AP   : {best_fold_AP:.4f}")
    print(f"Best LOC  : {best_fold_LOC:.4f}")

    

# =========================
# === 9. MAIN EXECUTION ===
# =========================
def main():
    set_seed(CFG.seed)
    df = pd.read_csv(CFG.files)
    run_kfold_training(df, train_transforms, val_transforms, n_splits=CFG.n_splits)

if __name__ == "__main__":
    main()

