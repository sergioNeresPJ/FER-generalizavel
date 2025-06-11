import os
from PIL import Image
from torch.utils.data import Dataset

class FERPlusDataset(Dataset):
    

    def __init__(self, root_dir, idxs_fer, idxs_test, subset='train', transform=None):
        """
        Args:
            root_dir (str): Caminho para o diretório raiz contendo as pastas train/validation/test.
            subset (str): Subconjunto do dataset, por exemplo: 'train', 'validation', ou 'test'.
            transform (callable, optional): Transformações a serem aplicadas nas imagens.
        """
        self.root_dir = os.path.join(root_dir, subset)
        self.transform = transform
        self.samples = []
        self.idxs_fer = idxs_fer
        self.idxs_test = idxs_test
        self.emotion_to_index = {
            "neutral": 0,
            "happy": 1,
            "surprise": 2,
            "sad": 3,
            "angry": 4,
            "disgust": 5,
            "fear": 6
        }

        for emotion, idx in self.emotion_to_index.items():
            emotion_dir = os.path.join(self.root_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            for img_name in os.listdir(emotion_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(emotion_dir, img_name)
                    self.samples.append((img_path, idx))

        print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.idxs_test[self.idxs_fer[label]]
        
        return image, label
