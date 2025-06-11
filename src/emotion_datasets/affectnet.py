import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd # Para o caso de annotations serem CSV/TXT também

class AffectNetDataset(Dataset):
    def __init__(self, root_dir, idxs_aff, idxs_test, subset='train', transform=None, annotation_type='npy'):
        """
        Inicializa o Dataset AffectNet.

        Args:
            root_dir (str): Caminho para o diretório raiz do AffectNet (contendo train_set e val_set).
            subset (str): 'train' para train_set ou 'val' para val_set.
            transform (callable, optional): Transformações a serem aplicadas nas imagens.
            annotation_type (str): Tipo de arquivo de anotação ('npy', 'csv', etc.).
                                   Este exemplo foca em 'npy'.
        """
        self.root_dir = root_dir
        self.subset_dir = os.path.join(root_dir, f"{subset}_set")
        self.images_dir = os.path.join(self.subset_dir, "images")
        self.annotations_dir = os.path.join(self.subset_dir, "annotations")
        self.transform = transform
        self.annotation_type = annotation_type

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Diretório de imagens não encontrado: {self.images_dir}")
        if not os.path.exists(self.annotations_dir):
            raise FileNotFoundError(f"Diretório de anotações não encontrado: {self.annotations_dir}")

        self.image_filenames = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))])

        # Opcional: Filtrar imagens que não possuem anotações correspondentes
        # Esta parte é importante para garantir que cada imagem tem uma anotação válida
        self.data_items = []
        for img_name in self.image_filenames:
            base_name = os.path.splitext(img_name)[0] # Remove extensão .jpg ou .png
            
            annotation_file = None
            if self.annotation_type == 'npy':
                annotation_file = os.path.join(self.annotations_dir, f"{base_name}_exp.npy")
            
            annotations = np.load(annotation_file)
            emotion_label = annotations.astype(int)

            if annotation_file and os.path.exists(annotation_file):
                if emotion_label < 7:
                    self.data_items.append((img_name, annotation_file))
            else:
                print(f"Aviso: Anotação não encontrada para a imagem {img_name}. Pulando.")

        if not self.data_items:
            raise ValueError(f"Nenhum par imagem-anotação encontrado no diretório {self.subset_dir}.")

        print(f"Carregado {len(self.data_items)} itens do {subset}_set.")
        
        self.idxs_aff = idxs_aff
        self.idxs_test = idxs_test

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name, annotation_file = self.data_items[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # 1. Carregar Imagem
        image = Image.open(img_path).convert('RGB') # Garante 3 canais de cor

        # 2. Carregar Anotações
        annotations = None
        if self.annotation_type == 'npy':
            
            annotations = np.load(annotation_file)
            
            emotion_label = annotations.astype(int).item()
            
            emotion_label = self.idxs_test[self.idxs_aff[emotion_label]]

            annotations = torch.tensor(emotion_label, dtype=torch.long)

        # 3. Aplicar Transformações
        if self.transform:
            image = self.transform(image)

        # Retorna a imagem e as anotações
        return image, annotations