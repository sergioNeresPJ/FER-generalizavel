import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random
import cv2
import numpy as np


class FERPlusDataset(Dataset):
    def __init__(self, root_dir, idxs_fer, idxs_test, subset="FER2013Train", transform=None):
        """
        Classe para lidar com o dataset FERPlus.

        Args:
            root_dir (str): Diretório raiz do dataset (ex: 'FER2013Plus').
            subset (str): Subconjunto a ser usado ('FER2013Train', 'FER2013Test', 'FER2013Valid').
            transform (callable, optional): Transformações para aplicar nas imagens.
        """
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.idxs_fer = idxs_fer
        self.idxs_test = idxs_test

        # Caminhos para imagens e labels
        self.images_dir = os.path.join(root_dir, "Images", subset)
        self.labels_path = os.path.join(root_dir, "Labels", subset, "label.csv")

        # Carregar o arquivo de labels
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Arquivo de labels não encontrado: {self.labels_path}")

        self.columns = [
            "image_name", "format", "neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt", "unknown", "NF"
        ]

        self.labels = pd.read_csv(self.labels_path, header=None, names=self.columns)

        # Validar se os arquivos de imagem existem
        self.image_files = self.labels['image_name']

        # Dicionário para mapear emoções para índices
        self.emotion_to_index = {
            "neutral": 0,
            "happiness": 1,
            "surprise": 2,
            "sadness": 3,
            "anger": 4,
            "disgust": 5,
            "fear": 6
        }

    def get_single_label_filtered(self, row):
        """
        Obtém o índice do rótulo mais votado entre as emoções, excluindo "unknown" e "NF".

        Args:
            row (pd.Series): Linha do DataFrame de rótulos.

        Returns:
            int: Índice do rótulo mais votado.
        """
        # Filtrar rótulos "unknown" e "NF"
        emotion_columns = ["neutral", "happiness", "surprise", "sadness",
                           "anger", "disgust", "fear"]
        # Obter o nome do rótulo mais votado
        emotion_name = row[emotion_columns].idxmax()
        # Retornar o índice correspondente
        return self.emotion_to_index[emotion_name]

    def __len__(self):
        #return 1
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise NotImplementedError("Slices não são suportados nesta implementação.")

        # Obter caminho da imagem
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Carregar imagem
        image = Image.open(img_path).convert("RGB")

        # Aplicar transformações se existirem
        if self.transform:
            image = self.transform(image)
        
        # Obter o rótulo correspondente
        label_row = self.labels.iloc[idx]
        single_label = self.get_single_label_filtered(label_row)

        single_label = self.idxs_test[self.idxs_fer[single_label]]

        return image, single_label