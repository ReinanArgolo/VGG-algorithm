import torch
from models.vgg import vgg16_bn
from torchsummary import summary # Se instalamos via pip/conda

def test_vgg_architecture():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Testando modelo no dispositivo: {device}")

    # Instancia VGG16
    model = vgg16_bn(num_classes=10).to(device)
    
    # Cria um batch falso de dados (Batch=2, Canais=3, H=32, W=32)
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    
    # Forward pass
    try:
        output = model(dummy_input)
        print(f"[SUCESSO] Forward pass concluído.")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape} (Esperado: [2, 10])")
        
        # Verifica assertiva
        assert output.shape == (2, 10)
        print("[SUCESSO] Shapes corretos!")
        
    except Exception as e:
        print(f"[ERRO] Falha no forward pass: {e}")
        return

    # Visualizar contagem de parâmetros
    print("\n--- Resumo do Modelo ---")
    try:
        summary(model, (3, 32, 32))
    except Exception as e:
        print("Biblioteta torchsummary não encontrada ou erro na exibição.")

if __name__ == "__main__":
    test_vgg_architecture()