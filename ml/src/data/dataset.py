"""
Dataset utilities and transforms for HAM10000 skin lesion dataset.
Used by all RQ notebooks for data loading.
"""
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from PIL import Image


def get_transforms(mode: str = 'val', img_size: int = 224):
    """
    Get albumentations transforms for training or validation.

    Args:
        mode: 'train' for augmentation, 'val' or 'test' for resize only
        img_size: target image size

    Returns:
        albumentations.Compose
    """
    if mode == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Lambda(name="to_tensor", image=lambda x, **kwargs: ToTensorV2()(image=x)["image"]),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Lambda(name="to_tensor", image=lambda x, **kwargs: ToTensorV2()(image=x)["image"]),
        ])


def create_splits(df: pd.DataFrame, seed: int = 42):
    """
    Create patient-level train/val/test splits (65/15/20).

    Args:
        df: metadata DataFrame with 'patient_id' column

    Returns:
        (train_df, val_df, test_df)
    """
    np.random.seed(seed)
    patients = df['patient_id'].unique()
    np.random.shuffle(patients)
    n_pts = len(patients)
    train_pts = set(patients[:int(n_pts * 0.65)])
    val_pts   = set(patients[int(n_pts * 0.65):int(n_pts * 0.80)])
    test_pts  = set(patients[int(n_pts * 0.80):])

    df = df.copy()
    df['_split'] = df['patient_id'].apply(
        lambda p: 'train' if p in train_pts else ('val' if p in val_pts else 'test')
    )
    train_df = df[df['_split'] == 'train']
    val_df   = df[df['_split'] == 'val']
    test_df  = df[df['_split'] == 'test']
    return train_df, val_df, test_df


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000 skin lesion images.
    """
    def __init__(self, split_df, img_size: int = 224, augment: bool = False, transform=None):
        self.df = split_df.dropna(subset=['filepath']).reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment
        self.transform = transform if transform else get_transforms('train' if augment else 'val', img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row['filepath']).convert('RGB'))
        augmented = self.transform(image=img)
        img_tensor = augmented['image']
        label = torch.tensor(row['label'], dtype=torch.float32)
        return img_tensor, label