import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class RafDataSet(Dataset):
    """
    Dataset para carregar o dataset RAF-DB.
    """

    def __init__(self, root_dir, idxs_raf, idxs_test, train=False, transform=None):
        """
        Args:
            root_dir (string): Diretório com todas as imagens.
            csv_file (string): Path para o arquivo CSV com as anotações.
            transform (callable, optional): Transformações a serem aplicadas em uma amostra.
        """
        self.root_dir = root_dir
        self.transform = transform
        if train==True:
            self.image_dir = os.path.join(self.root_dir, 'DATASET/train')
            csv_file = os.path.join(self.root_dir, 'train_labels.csv')
            self.annotations = pd.read_csv(csv_file)
        else:
            self.image_dir = os.path.join(self.root_dir, 'DATASET/test')
            csv_file = os.path.join(self.root_dir, 'test_labels.csv')
            self.annotations = pd.read_csv(csv_file)
        self.idxs_raf = idxs_raf
        self.idxs_test = idxs_test
            

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = int(self.annotations.iloc[idx, 1])

        img_name = os.path.join(self.image_dir, str(label), self.annotations.iloc[idx, 0])

        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.idxs_test[self.idxs_raf[label - 1]]

        return image, label