import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class FERPlusDataset(Dataset):
    """
    Classe de Dataset para o desafio Kaggle de Reconhecimento de Expressão Facial,
    com tratamento de erros robusto.
    """
    def __init__(self, csv_path, idxs_fer, idxs_test, train=True, transform=None):
        """
        Args:
            csv_path (string): Caminho para o arquivo fer2013.csv.
            idxs_fer (dict): Dicionário para o primeiro mapeamento de rótulos.
            idxs_test (dict): Dicionário para o segundo mapeamento de rótulos.
            train (bool): Se True, carrega o conjunto de 'Training'. Senão, 'PublicTest'.
            transform (callable, optional): Transformações a serem aplicadas.
        """
        # --- 1. Validação do caminho e leitura do arquivo ---
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Erro: O arquivo CSV não foi encontrado no caminho: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.replace(' ', '')
            self.df = df
        except Exception as e:
            raise IOError(f"Erro ao ler o arquivo CSV em '{csv_path}'. Detalhes: {e}")
        
        # --- 2. Validação das colunas do DataFrame ---
        required_columns = ['Usage', 'emotion', 'pixels']
        if not all(col in self.df.columns for col in required_columns):
            missing = set(required_columns) - set(self.df.columns)
            raise ValueError(f"Erro: Colunas essenciais faltando no CSV: {list(missing)}")

        # --- 3. Filtragem por 'Usage' e validação de dados ---
        usage = 'Training' if train else 'PublicTest'
        self.df = self.df[self.df['Usage'] == usage]
        if self.df.empty:
            raise ValueError(f"Erro: Nenhum dado encontrado para a categoria 'Usage' = '{usage}'. "
                             "Verifique o conteúdo do arquivo CSV.")
        
        # Reseta o índice do DataFrame após a filtragem para garantir acesso sequencial
        self.df = self.df.reset_index(drop=True)

        # --- 4. Validação da transformação ---
        if transform and not callable(transform):
            raise TypeError("Erro: O argumento 'transform' deve ser um objeto 'callable' (como uma função ou classe de transformação).")
        self.transform = transform
        
        # Armazena os dados e mapeamentos
        self.labels = self.df['emotion'].values
        self.pixels = self.df['pixels'].tolist()
        self.idxs_fer = idxs_fer
        self.idxs_test = idxs_test

    def __len__(self):
        """ Retorna o número total de amostras no dataset. """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Carrega e retorna uma amostra do dataset no índice `idx`.
        Amostra consiste em uma imagem e seu respectivo rótulo.
        """
        # --- 5. Validação do índice (embora o DataLoader geralmente previna isso) ---
        if not 0 <= idx < len(self.df):
            raise IndexError(f"Erro: Índice {idx} está fora do intervalo do dataset (Tamanho: {len(self.df)}).")

        try:
            # --- 6. Processamento seguro dos pixels ---
            img_pixels_str = self.pixels[idx].split(' ')
            
            # Verifica se o número de pixels é o esperado (48x48 = 2304)
            if len(img_pixels_str) != 2304:
                raise ValueError(f"O número de pixels ({len(img_pixels_str)}) não corresponde ao esperado (2304).")
            
            # Converte para numpy array, tratando exceções de conversão
            try:
                img_pixels = np.array(img_pixels_str, dtype='uint8')
            except ValueError:
                raise ValueError("Contém valores não numéricos na string de pixels.")
            
            img = img_pixels.reshape(48, 48)
            img = Image.fromarray(img)
            
            # Se a imagem for em escala de cinza (modo 'L'), converte para RGB
            if img.mode == 'L':
                img = img.convert('RGB') # Isso replicará o canal L em R, G e B
            # -----------------------------------------------

            # Aplica transformações
            if self.transform:
                img = self.transform(img)
            
            # --- 7. Processamento seguro dos rótulos ---
            original_label = self.labels[idx]
            
            try:
                # Aplica o mapeamento em duas etapas
                mapped_label_step1 = self.idxs_fer[original_label]
                final_label = self.idxs_test[mapped_label_step1]
                label = torch.tensor(final_label).long()
            except KeyError as e:
                # Erro mais provável se a label não estiver no dicionário
                raise KeyError(f"O rótulo '{e.args[0]}' do CSV não foi encontrado nos dicionários de mapeamento.")
            except IndexError:
                # Caso os mapeamentos usem listas/tuplas em vez de dicionários
                raise IndexError(f"O rótulo '{original_label}' do CSV resultou em um índice fora dos limites para os mapeamentos.")

            return img, label

        except (ValueError, IndexError, KeyError, TypeError) as e:
            # Captura qualquer erro de processamento para este item específico
            print(f"AVISO: Falha ao carregar a amostra no índice {idx}. Motivo: {e}")
            print("Retornando a próxima amostra válida ou None se for a última.")
            # Estratégia de recuperação: tentar carregar o próximo item
            # Evita que um único item corrompido quebre todo o processo de treinamento
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                # Se não houver próximo item, é preciso que o 'collate_fn' do DataLoader saiba lidar com None
                return None, None