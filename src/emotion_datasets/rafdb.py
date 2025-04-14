import torch.utils.data as data
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random
import cv2
import numpy as np



class RafDataSet(data.Dataset):
    def __init__(self, raf_path, idxs_raf, idxs_test, dataidxs=None, train=True, transform=None, basic_aug=False, download=False):
        self.train = train
        self.dataidxs = dataidxs
        self.transform = transform
        self.raf_path = raf_path
        self.idxs_raf = idxs_raf
        self.idxs_test = idxs_test

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if self.train:
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.target = dataset.iloc[:, LABEL_COLUMN].astype(int).values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        self.target = np.array(self.target)

        self.file_paths = []
        for f in file_names:    # use raf-db aligned images for training/testing
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        ################
        self.file_paths = np.array(self.file_paths)
        if self.dataidxs is not None:
            self.file_paths = self.file_paths[self.dataidxs]
            self.target = self.target[self.dataidxs]
        else:
            self.file_paths = self.file_paths
        self.file_paths = self.file_paths.tolist()


    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.target

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        sample = cv2.imread(path)
        sample = sample[:, :, ::-1]  # BGR to RGB (Optional)
        target = self.target[idx]

        target = self.idxs_test[self.idxs_raf[target]]
        
        if self.transform is not None:
            
            sample = Image.fromarray(sample.copy())  # Convert NumPy array to PIL image
            sample = self.transform(sample)
        
        return sample, target  # , idx (Optional to return index)
