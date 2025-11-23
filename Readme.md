# Classificação de Tipos de Solo com Transfer Learning (PyTorch)

Este projeto aplica técnicas de **Deep Learning** e **Transfer Learning** para classificar diferentes tipos de solo a partir de imagens. O código utiliza a biblioteca **PyTorch** para treinar e avaliar dois modelos de arquiteturas consagradas: **AlexNet** e **ResNet18**.

## 📌 Sobre o Projeto

O objetivo deste notebook é criar um classificador de imagens capaz de identificar 7 tipos diferentes de solos, auxiliando em análises agrícolas e geológicas. O projeto utiliza uma abordagem de *Hold-out* para divisão dos dados e *Transfer Learning* (congelamento de pesos das camadas convolucionais) para adaptar modelos pré-treinados ao novo dataset.

## 📂 Dataset

O conjunto de dados utilizado é o **Comprehensive Soil Classification Dataset**, obtido via API do Kaggle.

* **Fonte:** [Kaggle - Comprehensive Soil Classification Datasets](https://www.kaggle.com/datasets/ai4a-lab/comprehensive-soil-classification-datasets)
* **Classes (7 tipos):**
    * Alluvial_Soil (Solo Aluvial)
    * Arid_Soil (Solo Árido)
    * Black_Soil (Solo Preto)
    * Laterite_Soil (Solo Laterítico)
    * Mountain_Soil (Solo de Montanha)
    * Red_Soil (Solo Vermelho)
    * Yellow_Soil (Solo Amarelo)

## 🛠️ Tecnologias e Bibliotecas

* **Linguagem:** Python 3
* **Framework de DL:** PyTorch, Torchvision
* **Processamento de Dados:** Numpy, PIL (Pillow), Glob
* **Métricas e Split:** Scikit-learn
* **Visualização:** Matplotlib, Seaborn
* **Ambiente:** Google Colab

## 🧠 Metodologia

1.  **Preparação dos Dados:**
    * Download automático via Kaggle API.
    * Divisão dos dados: Treino (60%), Validação (20%) e Teste (20%).
    * **Transformações:** Redimensionamento para 224x224, *Data Augmentation* (RandomHorizontalFlip) para o treino e Normalização (baseada nas médias da ImageNet).

2.  **Arquiteturas de Modelos:**
    * **AlexNet:** Pré-treinada. Camadas de *features* congeladas. Camada final ajustada para 7 saídas.
    * **ResNet18:** Pré-treinada. Parâmetros congelados. Camada `fc` (fully connected) substituída.

3.  **Treinamento:**
    * **Função de Perda:** CrossEntropyLoss.
    * **Otimizador:** SGD (Stochastic Gradient Descent) com Momentum.
    * **Épocas:** 50 épocas para cada modelo.
    * **Batch Size:** 32.

## 📊 Resultados

Abaixo estão as métricas obtidas na avaliação do conjunto de teste (dados nunca vistos pelo modelo durante o treino):

| Modelo | Acurácia no Teste | Melhor Acurácia (Validação) |
| :--- | :---: | :---: |
| **AlexNet** | **86.55%** | 90.72% |
| **ResNet18** | 82.35% | 85.23% |

### Performance por Classe (Exemplo AlexNet)
O modelo obteve excelente desempenho em solos como **Black_Soil (F1: 0.98)** e **Yellow_Soil (F1: 0.89)**, mas apresentou maior dificuldade em distinguir **Alluvial_Soil**.

## 🚀 Como Executar

1.  Clone este repositório.
2.  Certifique-se de ter uma conta no Kaggle e um token de API (`kaggle.json`).
3.  Abra o notebook no Google Colab ou Jupyter Notebook local.
4.  Instale as dependências necessárias:
    ```bash
    pip install torch torchvision scikit-learn matplotlib seaborn
    ```
5.  Carregue o arquivo `kaggle.json` quando solicitado na primeira célula para baixar o dataset.
6.  Execute as células sequencialmente.

## 📈 Visualizações

O notebook gera os seguintes gráficos para análise:
* Curvas de Loss (Treino vs Validação).
* Curvas de Acurácia (Treino vs Validação).
* Matriz de Confusão (Heatmap) para análise de erros entre classes.

---
*Desenvolvido como parte de estudos em Visão Computacional e Deep Learning.*