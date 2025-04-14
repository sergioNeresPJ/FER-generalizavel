# 🧠 Emotion Datasets – PyTorch Dataset Wrappers

Este repositório contém implementações de datasets prontos para uso com PyTorch, focados em reconhecimento de emoções faciais. Em vez de armazenar os dados em si (imagens JPEG/PNG), este repositório fornece **classes que herdam de `torch.utils.data.Dataset`**, encapsulando a lógica necessária para carregar e preparar os dados a partir de caminhos locais previamente baixados.

## 📦 Datasets disponíveis

O repositório implementa wrappers para os seguintes conjuntos de dados populares:

| Dataset        | Arquivo             | Emoções Usadas | Fonte do Dataset(Kaggle) |
|----------------|---------------------|---------|--------------------------|
| AffectNet      | `affectnet.py`      | 7       | [AffectNet](https://www.kaggle.com/datasets/yakhyokhuja/affectnetaligned)|
| SFEW           | `sfew.py`           | 7       | [SFEW](https://www.kaggle.com/datasets/vlntnstarodub/datasetsfew) |
| FER+           | `ferplus.py`        | 7       | [FER+](https://www.kaggle.com/datasets/ss1033741293/ferplus) |
| M&MA (M3A)     | `mma.py`            | 7       | [MMA](https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression) |
| RAF-DB         | `rafdb.py`          | 7       | [RAF-DB](https://www.kaggle.com/datasets/kuryakin/eacdata/data)  |

## 📁 Estrutura do Repositório

```
datasets/ 
├── affectnet.py # Dataset AffectNet 
├── sfew.py # Dataset SFEW 
├── ferplus.py # Dataset FER+ 
├── mma.py # Dataset M&MA (customizado) 
├── rafdb.py # Dataset RAF-DB 
└── README.md # Esta documentação
```

Cada classe implementa os seguintes métodos obrigatórios:
- `__init__()`: Inicialização do dataset e carregamento de metadados.
- `__getitem__(index)`: Retorna um item do dataset (imagem, rótulo).
- `__len__()`: Retorna o tamanho do dataset.

# 📌 Observações

Certifique-se de respeitar os termos de uso de cada dataset ao baixá-los e utilizá-los.

# 📬 Contato

Para dúvidas ou sugestões, sinta-se à vontade para abrir uma issue ou enviar um pull request.

