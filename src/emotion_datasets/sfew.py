import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random
import cv2
import numpy as np

class SFEWDataset(Dataset):
    def __init__(self, root_dir, idx_test, idx_sfew, split="Train", transform=None):
        """
        Args:
            root_dir (str): Diretório raiz contendo as pastas train, test e valid.
            split (str): Qual partição carregar ("train", "test" ou "valid").
            transform (callable, optional): Transformações a serem aplicadas às imagens.
        """
        self.root_dir = os.path.join(root_dir, split, "Test_Aligned_Faces") if split == "Test" else os.path.join(root_dir, split)
        print(self.root_dir)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))  # Lista de emoções
        self.idxs_test = idx_test
        self.idxs_sfew = idx_sfew
        
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):  # Filtra apenas imagens
                    self.samples.append((os.path.join(class_path, filename), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        label = self.idxs_test[self.idxs_sfew[label]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label