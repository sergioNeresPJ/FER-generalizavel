import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random
import cv2
import numpy as np

class AffectNetDataset(Dataset):
    def __init__(self, root_dir, idx_aff, idx_test, split="train", transform=None):
        """
        Args:
            root_dir (str): Diretório raiz contendo as pastas train, test e valid.
            split (str): Qual partição carregar ("train", "test" ou "valid").
            transform (callable, optional): Transformações a serem aplicadas às imagens.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.idxs_test = idx_test
        self.idxs_aff = idx_aff
        
        self.class_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happiness": 3,
            "sadness": 4,
            "surprise": 5,
            "neutral": 6
        }
        
        self.samples = []
        for class_idx in range(len(self.class_map)):  # Pastas nomeadas por índice
            class_path = os.path.join(self.root_dir, str(class_idx))
            if os.path.exists(class_path):
                for filename in os.listdir(class_path):
                    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Filtra apenas imagens
                        self.samples.append((os.path.join(class_path, filename), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        label = self.idxs_test[self.idxs_aff[label]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label