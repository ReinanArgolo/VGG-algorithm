import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

NUM_WORKERS = 0
PIN_MEMORY = True

def get_transforms(): 
    """Define as trasnformaçoes para treino e teste."""

    # Transformações mais agressivas para treino (Data Augmentation)
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Apenas normalização para validação/teste
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])
    
    return train_transforms, test_transform


def create_dataloaders(batch_size = 64, data_dir='data/downloaded'):
    """
        Baixa o dataset e cria o dataloaders.
        Retorna: train_loader, val_loader, test_loader, classes
    """


    train_transform, test_transform = get_transforms()

    print(f"[INFO] Baixando/carregando CIFAR-10 em {data_dir}")

    #baixar Dataset Completo

    train_dataset_full = datasets.CIFAR10(root=data_dir, train=True,
                                          download=True, transform=train_transform)
    
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)
    
    # Dividir treino (45k) e Validação (5k)
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size

    train_subset, val_subset = random_split(train_dataset_full, [train_size, val_size])

    # Criar Loaders

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                               shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                               shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    classes = train_dataset_full.classes

    return train_loader, val_loader, train_loader, classes

if __name__ == "__main__":
    
    # Testar a criação dos dataloaders
    tl, vl, tsl, classes = create_dataloaders()


    images, labels = next(iter(tl))

    print(f"Shape do batch de imagens: {images.shape} (Batch Size, Canais, Altura, Largura)")
    print(f"Shape do batch de labels: {labels.shape} (Batch Size)")
