import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """Fixa as sementes para garantir reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.determitisc = True
    torch.backends.cudnn.benchmark = False 

    print(f"[INFG] Seed Fixada em: {seed}")


def get_device():
    """Retorna o dispositivo disponível CPU ou GPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo Selecioando: {device}")

    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    
    return device


    