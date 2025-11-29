import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # Barra de progresso bonita
import sys
import os
import time

# Adiciona a raiz do projeto ao path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vgg import vgg16_bn
from data.data_setup import create_dataloaders
from utils.reproducibility import set_seed, get_device

# --- HIPERPARÂMETROS ---
BATCH_SIZE = 64 # Se der erro de memória na GPU, diminua para 32
LR = 0.01       # Learning Rate (SGD costuma gostar de 0.01 ou 0.001)
EPOCHS = 10     # Vamos começar com 10 para testar. O ideal seria 30-50.
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4 # Regularização L2 (importante para VGG não overfittar)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # Habilita Dropout e BatchNorm
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="Treinando", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        # 1. Zerar gradientes anteriores
        optimizer.zero_grad()
        
        # 2. Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 3. Backward Pass
        loss.backward()
        
        # 4. Otimização
        optimizer.step()
        
        # Métricas
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item())
        
    avg_loss = running_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

def validate(model, loader, criterion, device):
    model.eval() # Desabilita Dropout e trava BatchNorm
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # Não calcula gradientes (economiza memória)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

def main():
    set_seed(42)
    device = get_device()
    
    # 1. Dados
    train_loader, val_loader, _, classes = create_dataloaders(batch_size=BATCH_SIZE)
    
    # 2. Modelo
    print("[INFO] Criando VGG-16...")
    model = vgg16_bn(num_classes=10).to(device)
    
    # 3. Loss e Otimizador
    criterion = nn.CrossEntropyLoss()
    
    # SGD com Momentum é o clássico para treinar VGG
    optimizer = optim.SGD(model.parameters(), lr=LR, 
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # Scheduler: Reduz o LR quando a loss para de cair (Platô)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # 4. Loop Principal
    best_acc = 0.0
    print(f"\n[INFO] Iniciando treino por {EPOCHS} épocas...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"\n--- Época {epoch+1}/{EPOCHS} ---")
        
        # Treino
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validação
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Atualiza LR
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Checkpoint: Salva o melhor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), 'checkpoints/vgg16_cifar10_best.pth')
            print("Melhor modelo salvo!")

    total_time = time.time() - start_time
    print(f"\n[FIM] Treino finalizado em {total_time/60:.2f} minutos.")
    print(f"Melhor Acurácia na Validação: {best_acc:.2f}%")

if __name__ == "__main__":
    main()