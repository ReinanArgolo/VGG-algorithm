from utils.reproducibility import set_seed, get_device
import torch

def smoke_test():

    set_seed(42)
    device = get_device()

    x = torch.randn(3,3).to(device)
    y = torch.matmul(x,x)

    print("\n[SUCESSO] Ambiente configurado corretamente!")
    print(f"Resultado do teste de tensor:\n{y}")

if __name__ == "__main__":
    smoke_test()