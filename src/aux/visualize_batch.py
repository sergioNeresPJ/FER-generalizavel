import matplotlib.pyplot as plt
import torch

def visualize_batch(train_loader, index_to_emotion, num_rows=4, num_cols=8):
    """
    Visualiza um batch de imagens do DataLoader em um grid, mostrando seus rótulos.

    Args:
        train_loader (torch.utils.data.DataLoader): O DataLoader contendo as imagens e rótulos.
        index_to_emotion (dict): Um dicionário que mapeia índices de rótulos para nomes de emoções.
        num_rows (int): O número de linhas no grid de visualização.
        num_cols (int): O número de colunas no grid de visualização.
    """
    # Obtendo o batch de imagens e rótulos
    for images, labels in train_loader:
        # Se você quiser mostrar apenas um batch
        break

    # Definindo o layout para as linhas e colunas especificadas
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))
    axes = axes.flatten()  # Flatten para facilitar a iteração

    # Loop para exibir as imagens no grid
    for i, (img, label) in enumerate(zip(images, labels)):
        if i >= len(axes):  # Se houver mais imagens do que subgráficos
            break

        # Convertendo a imagem para numpy e normalizando
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Exibindo a imagem
        ax = axes[i]
        ax.imshow(img_np)
        ax.axis('off')  # Desativar os eixos

        # Usando o mapa de rótulos para mostrar o nome da emoção
        label_name = index_to_emotion[label.item()]
        ax.set_title(f"{label_name}", fontsize=10)  # Título com o nome do label

    # Ajustar o layout para não sobrepor as imagens
    plt.tight_layout()
    plt.show()