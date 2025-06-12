import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FERPlusDataset(Dataset):
    """
    Um PyTorch Dataset para o conjunto de dados FERPLUS.

    Args:
        root_dir (str): O diretório raiz onde os subdiretórios FER2013Test,
                        FER2013Train e FER2013Valid estão localizados.
        csv_file (str): O caminho completo para o arquivo 'fer2013new.csv'.
        idxs_fer (dict): Dicionário de mapeamento de índices FER+ para um formato intermediário.
                         Ex: {0: 0, 1: 1, ...}
        idxs_test (dict): Dicionário de mapeamento do formato intermediário para o rótulo final.
                         Ex: {0: 0, 1: 1, ...}
        subset (str): O subconjunto de dados a ser carregado ('train', 'valid' ou 'test').
        transform (callable, optional): Uma função/transformação a ser aplicada nas imagens.
    """
    def __init__(self, root_dir: str, csv_file: str, idxs_fer: dict, idxs_test: dict, subset: str, transform=None):
        if not isinstance(root_dir, str) or not os.path.isdir(root_dir):
            raise ValueError(f"O parâmetro 'root_dir' deve ser um caminho de diretório válido. '{root_dir}' não é um diretório.")
        
        if not isinstance(csv_file, str) or not os.path.exists(csv_file):
            raise ValueError(f"O parâmetro 'csv_file' deve ser um caminho de arquivo CSV válido. '{csv_file}' não é um arquivo ou não existe.")
            
        if subset not in ['train', 'valid', 'test']:
            raise ValueError("O parâmetro 'subset' deve ser 'train', 'valid' ou 'test'.")

        if not isinstance(idxs_fer, dict):
            raise TypeError("O parâmetro 'idxs_fer' deve ser um dicionário.")
        if not isinstance(idxs_test, dict):
            raise TypeError("O parâmetro 'idxs_test' deve ser um dicionário.")

        self.root_dir = root_dir
        self.csv_file = csv_file
        self.subset = subset
        self.transform = transform
        
        # Definir as colunas de emoção esperadas no CSV (do fer2013new.csv)
        self.raw_column_names = [
            'usage', 'Image name', 'neutral', 'happiness', 'surprise', 'sadness', 
            'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF'
        ]
        
        # As 7 emoções que você quer focar para o mapeamento
        self.emotion_cols = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        self.emotion_labels = self.emotion_cols # Usamos a mesma lista para mapeamento de índice

        self.idxs_fer = idxs_fer
        self.idxs_test = idxs_test

        self.data_frame = self._load_data()
        
        if self.__len__() == 0:
            raise RuntimeError(f"Nenhum dado válido encontrado para o subconjunto '{self.subset}' no arquivo CSV '{self.csv_file}'. Verifique o conteúdo e a estrutura do CSV e do diretório de imagens.")

    def _load_data(self):
        """
        Carrega o arquivo fer2013new.csv, prepara o DataFrame e processa os rótulos.
        Associa as imagens com base na coluna 'Image name' do CSV e filtra
        pelo 'usage' para o subset correto.
        """
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"O arquivo CSV não foi encontrado em: {self.csv_file}. Certifique-se de que o caminho está correto.")
        
        try:
            df_full = pd.read_csv(self.csv_file, header=0, names=self.raw_column_names) # header=0 pois tem cabeçalho
        except pd.errors.EmptyDataError:
            raise ValueError(f"O arquivo CSV '{self.csv_file}' está vazio.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Erro ao analisar o arquivo CSV '{self.csv_file}'. Verifique a formatação do arquivo. Detalhe: {e}")
        except Exception as e:
            raise RuntimeError(f"Ocorreu um erro inesperado ao carregar o CSV '{self.csv_file}': {e}")

        # Verificar se todas as colunas esperadas estão presentes
        for col in self.raw_column_names:
            if col not in df_full.columns:
                raise RuntimeError(f"A coluna esperada '{col}' não foi encontrada no CSV '{self.csv_file}'. Verifique os nomes das colunas e o cabeçalho.")

        # Mapeamento do 'usage' do CSV para o nome da pasta
        usage_to_subset_map = {
            'Training': 'train',
            'PublicTest': 'valid', 
            'PrivateTest': 'test' 
        }

        # Filtrar o DataFrame pelo 'usage' correspondente ao subset
        csv_usage_name = None
        for usage_key, subset_val in usage_to_subset_map.items():
            if subset_val == self.subset:
                csv_usage_name = usage_key
                break
        
        if csv_usage_name is None:
            raise ValueError(f"Mapeamento interno de 'subset' para 'usage' no CSV está incorreto para '{self.subset}'.")

        df_subset = df_full[df_full['usage'] == csv_usage_name].copy()

        if df_subset.empty:
            return pd.DataFrame() # Retorna um DataFrame vazio se não houver dados

        subset_dir_map = {
            'train': 'FER2013Train',
            'valid': 'FER2013Valid',
            'test': 'FER2013Test'
        }
        image_folder = subset_dir_map[self.subset]
        
        # Assegurar que 'Image name' seja string e não tenha NaNs
        df_subset['Image name'] = df_subset['Image name'].fillna('').astype(str)
        
        # Filtrar linhas onde 'Image name' ficou vazio após fillna (indicando dados ausentes/inválidos)
        initial_image_name_count = len(df_subset)
        df_subset = df_subset[df_subset['Image name'] != '']
        if len(df_subset) < initial_image_name_count:
            pass # Removed print for clean output

        # Construir o caminho completo da imagem usando a coluna 'Image name'
        df_subset['image_path'] = df_subset['Image name'].apply(
            lambda img_name: os.path.join(self.root_dir, image_folder, img_name)
        )
        
        # Encontrar o rótulo com a maior contagem de votos para cada imagem
        def get_single_label(row):
            # Obtém as contagens de votos APENAS PARA AS EMOÇÕES DESEJADAS (as 7 principais)
            votes = row[self.emotion_cols].to_dict()
            
            if not votes: 
                return None
            
            numeric_votes = {k: pd.to_numeric(v, errors='coerce') for k, v in votes.items()}
            numeric_votes = {k: v for k, v in numeric_votes.items() if not pd.isna(v)}

            if not numeric_votes: 
                return None 
            
            # Use max(key=dict.get) para encontrar a chave com o valor máximo
            max_emotion = max(numeric_votes, key=numeric_votes.get)
            
            return max_emotion

        df_subset['single_label'] = df_subset.apply(get_single_label, axis=1)

        # Filtra linhas onde o caminho da imagem pode ser inválido
        initial_count = len(df_subset)
        df_subset = df_subset[df_subset['image_path'].apply(lambda x: os.path.exists(x))]
        if len(df_subset) < initial_count:
            pass # Removed print for clean output
            
        return df_subset

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.data_frame)):
            raise IndexError(f"Índice {idx} fora dos limites do dataset (tamanho: {len(self.data_frame)}).")

        img_path = self.data_frame.iloc[idx]['image_path']
        label_str = self.data_frame.iloc[idx]['single_label']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Imagem não encontrada em '{img_path}' para o índice {idx}. Por favor, verifique o arquivo.")
        except UnidentifiedImageError:
            raise UnidentifiedImageError(f"Não foi possível identificar o formato da imagem em '{img_path}' para o índice {idx}. Pode estar corrompida.")
        except Exception as e:
            raise RuntimeError(f"Erro inesperado ao carregar a imagem '{img_path}' para o índice {idx}: {e}")

        try:
            original_label_idx = self.emotion_labels.index(label_str)
        except ValueError:
            raise ValueError(f"Rótulo '{label_str}' para o índice {idx} não encontrado em 'emotion_labels'. Verifique a consistência dos dados.")
            
        try:
            # Apply the two-step mapping
            mapped_label_step1 = self.idxs_fer[original_label_idx]
            final_label = self.idxs_test[mapped_label_step1]
            label = torch.tensor(final_label).long()
        except KeyError as e:
            raise KeyError(f"Mapeamento de rótulo falhou para '{label_str}' (índice original: {original_label_idx}) no índice {idx}. Chave não encontrada no dicionário de mapeamento: {e.args[0]}.")
        except Exception as e:
            raise RuntimeError(f"Erro inesperado no mapeamento de rótulos para '{label_str}' no índice {idx}: {e}.")

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                raise RuntimeError(f"Erro ao aplicar transformação na imagem '{img_path}' para o índice {idx}: {e}.")
            
        return image, label