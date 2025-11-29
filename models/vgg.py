import torch
import torch.nn as nn

# Configurações padrão da família VGG
# 'M' significa MaxPool, números significam canais da Convolução
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, dropout=0.5):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        
        # 1. Cria a parte convolucional (Feature Extractor)
        self.features = self._make_layers(cfg[vgg_name])
        
        # 2. Cria a parte densa (Classifier)
        # Nota: Adaptado para CIFAR-10 (32x32). 
        # O output final das features será (512, 1, 1).
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        
        # Inicialização de pesos (opcional, mas ajuda a convergir)
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3 # RGB
        
        for x in cfg:
            if x == 'M':
                # MaxPool reduz dimensão pela metade
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # Conv3x3 mantém dimensão (padding=1)
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x), # BatchNorm acelera muito o treino!
                           nn.ReLU(inplace=True)]
                in_channels = x
                
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def vgg11_bn(num_classes=10): return VGG('VGG11', num_classes)
def vgg13_bn(num_classes=10): return VGG('VGG13', num_classes)
def vgg16_bn(num_classes=10): return VGG('VGG16', num_classes)
def vgg19_bn(num_classes=10): return VGG('VGG19', num_classes)