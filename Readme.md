# Classificação de Tipos de Solo com Transfer Learning (PyTorch)

Este projeto implementa técnicas de **Deep Learning** e **Transfer Learning** para classificar automaticamente diferentes tipos de solo a partir de imagens. Utilizando a biblioteca **PyTorch**, o projeto treina e avalia dois modelos de arquiteturas consagradas: **AlexNet** e **ResNet18**.

## Sobre o Projeto

O objetivo principal é desenvolver um classificador de imagens capaz de identificar **7 tipos diferentes de solos**, auxiliando em análises agrícolas e geológicas. O projeto emprega uma abordagem de *Hold-out* para divisão dos dados e *Transfer Learning* com congelamento de pesos das camadas convolucionais para adaptar modelos pré-treinados ao dataset de solos.

### Principais Características

- **Transfer Learning** com modelos pré-treinados na ImageNet
- **Hold-out Strategy**: Divisão estratificada em treino (60%), validação (20%) e teste (20%)
- **Data Augmentation** para melhorar a generalização
- **Reprodutibilidade** garantida através de seeds fixas
- **Visualizações** detalhadas de métricas e matriz de confusão

## Dataset

O conjunto de dados utilizado é o **Comprehensive Soil Classification Dataset**, disponível no Kaggle.

### Informações do Dataset

- **Fonte:** [Kaggle - Comprehensive Soil Classification Datasets](https://www.kaggle.com/datasets/ai4a-lab/comprehensive-soil-classification-datasets)
- **Total de Classes:** 7 tipos de solo
- **Formato:** Imagens JPG organizadas por pastas de classes

### Classes de Solo

| Classe | Descrição | Nome em Português |
|--------|-----------|-------------------|
| `Alluvial_Soil` | Solo formado por sedimentos depositados por rios | Solo Aluvial |
| `Arid_Soil` | Solo de regiões áridas e semi-áridas | Solo Árido |
| `Black_Soil` | Solo rico em matéria orgânica | Solo Preto |
| `Laterite_Soil` | Solo tropical rico em ferro e alumínio | Solo Laterítico |
| `Mountain_Soil` | Solo de regiões montanhosas | Solo de Montanha |
| `Red_Soil` | Solo com alta concentração de óxido de ferro | Solo Vermelho |
| `Yellow_Soil` | Solo com coloração amarelada | Solo Amarelo |

## Tecnologias e Bibliotecas

### Principais Dependências

```
torch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.21.0
matplotlib >= 3.5.0
seaborn >= 0.12.0
scikit-learn >= 1.0.0
Pillow >= 9.0.0
```

### Estrutura de Bibliotecas

- **Framework de Deep Learning:** PyTorch, Torchvision
- **Processamento de Dados:** NumPy, PIL (Pillow), Glob
- **Métricas e Validação:** Scikit-learn
- **Visualização:** Matplotlib, Seaborn
- **Ambiente Recomendado:** Google Colab (com GPU)

## Metodologia

### 1. Preparação dos Dados

#### Download do Dataset
- Download automático via **Kaggle API**
- Extração e organização das imagens por classe

#### Divisão dos Dados (Hold-out Strategy)
```
Total de Dados: 100%
├── Treino: 60%
├── Validação: 20%
└── Teste: 20%
```
- Divisão estratificada para manter a proporção de classes
- Seed fixa (42) para reprodutibilidade

#### Transformações de Dados

**Conjunto de Treino:**
- `RandomResizedCrop(224)`: Crop aleatório redimensionado
- `RandomHorizontalFlip()`: Inversão horizontal aleatória (Data Augmentation)
- `ToTensor()`: Conversão para tensor PyTorch
- `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`: Normalização ImageNet

**Conjuntos de Validação e Teste:**
- `Resize(256)`: Redimensionamento para 256px
- `CenterCrop(224)`: Crop central de 224x224
- `ToTensor()`: Conversão para tensor
- `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`: Normalização ImageNet

### 2. Arquiteturas de Modelos

#### AlexNet
- **Origem:** Pré-treinada na ImageNet
- **Estratégia:** Congelamento de todas as camadas de features
- **Modificação:** Substituição da última camada do classificador (6) por uma Linear com 7 saídas
- **Parâmetros Treináveis:** Apenas a camada final

#### ResNet18
- **Origem:** Pré-treinada na ImageNet
- **Estratégia:** Congelamento de todas as camadas convolucionais
- **Modificação:** Substituição da camada `fc` (fully connected) por uma Linear com 7 saídas
- **Parâmetros Treináveis:** Apenas a camada final

### 3. Configuração de Treinamento

| Hiperparâmetro | Valor |
|----------------|-------|
| **Função de Perda** | CrossEntropyLoss |
| **Otimizador** | SGD com Momentum |
| **Learning Rate** | 0.0001 |
| **Momentum** | 0.9 |
| **Épocas** | 50 |
| **Batch Size** | 32 |
| **Device** | CUDA (GPU) se disponível |

### 4. Processo de Treinamento

- **Early Stopping:** Salvamento do melhor modelo baseado na acurácia de validação
- **Monitoramento:** Loss e acurácia para treino e validação a cada época
- **Histórico:** Registro de todas as métricas para análise posterior

## Avaliação e Métricas

### Métricas Calculadas

- **Acurácia Geral:** Percentual de predições corretas
- **Precision, Recall e F1-Score:** Por classe
- **Matriz de Confusão:** Visualização detalhada dos erros de classificação

### Visualizações Geradas

1. **Gráficos de Treinamento:**
   - Loss (Treino vs Validação)
   - Acurácia (Treino vs Validação)
   - Learning Rate ao longo das épocas

2. **Matriz de Confusão:**
   - Visualização com heatmap
   - Salva como `confusion_matrix.png` (AlexNet)
   - Salva como `MatrizResNet.png` (ResNet18)

## Como Usar

### Pré-requisitos

1. **Conta no Kaggle** com API configurada
2. **Arquivo kaggle.json** com suas credenciais
3. **Google Colab** (recomendado) ou ambiente Python local com GPU

### Passo a Passo

#### 1. Configurar Kaggle API

```python
# Upload do arquivo kaggle.json no Colab
from google.colab import files
files.upload()
```

#### 2. Executar o Notebook

Execute as células sequencialmente:

1. **Setup Inicial:** Configuração da Kaggle API
2. **Download:** Download e extração do dataset
3. **Imports:** Importação de bibliotecas
4. **Preparação:** Divisão e transformação dos dados
5. **Treinamento AlexNet:** Treino do primeiro modelo
6. **Avaliação AlexNet:** Métricas e visualizações
7. **Treinamento ResNet18:** Treino do segundo modelo
8. **Avaliação ResNet18:** Métricas e visualizações

#### 3. Acessar Resultados

Os modelos treinados são salvos em:
- `./exp_alexnet/best_model_alexnet_soil.pt`
- `./exp_resnet/best_model_resnet18_soil.pt`

As matrizes de confusão são salvas como:
- `confusion_matrix.png`
- `MatrizResNet.png`

## Estrutura do Projeto

```
Trabalho-Sin-323/
│
├── Classificacao_Solos_CNN.ipynb    # Notebook principal
├── Readme.md                         # Este arquivo
├── .gitignore                        # Arquivos ignorados pelo Git
│
├── exp_alexnet/                      # Modelos AlexNet (gerado)
│   └── best_model_alexnet_soil.pt
│
├── exp_resnet/                       # Modelos ResNet (gerado)
│   └── best_model_resnet18_soil.pt
│
├── confusion_matrix.png              # Matriz AlexNet (gerado)
└── MatrizResNet.png                  # Matriz ResNet (gerado)
```

## Resultados Esperados

- **Acurácia no Conjunto de Teste:** Varia conforme o modelo e execução
- **Convergência:** Observável nos gráficos de loss e acurácia
- **Generalização:** Avaliada pela diferença entre treino e validação


## Reprodutibilidade

O projeto implementa medidas para garantir resultados reproduz�veis:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Refer�ncias

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Comprehensive Soil Classification Dataset](https://www.kaggle.com/datasets/ai4a-lab/comprehensive-soil-classification-datasets)
