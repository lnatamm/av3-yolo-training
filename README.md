# 🪓 YOLOv8 Beverages Detection - Treinamento e Detecção

Projeto de treinamento de modelo YOLOv8 para detecção de bebidas em imagens usando deep learning.

## 📋 Índice

- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Usar](#-como-usar)
- [Resultados](#-resultados)
- [Troubleshooting](#-troubleshooting)

## 🔧 Requisitos

### Hardware
- **GPU NVIDIA** (recomendado): GeForce RTX 3060 ou superior
- **RAM**: Mínimo 8GB (16GB recomendado)
- **Armazenamento**: 5GB livres

### Software
- **Windows 10/11**
- **Python 3.13+**
- **CUDA 12.4** (para suporte a GPU)
- **Git** (opcional)

## 📦 Instalação

### 1. Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd av3-yolo-training
```

### 2. Criar Ambiente Virtual

```bash
python -m venv venv
```

### 3. Ativar o Ambiente Virtual

**PowerShell:**
```powershell
.\venv\Scripts\activate
```

**CMD:**
```cmd
venv\Scripts\activate.bat
```

### 4. Instalar Dependências

#### Opção A: Usando o Script Automático (Recomendado)

Execute o script de instalação que configura tudo automaticamente:

```powershell
.\install.ps1
```

O script irá:
- ✅ Atualizar pip, setuptools e wheel
- 🔥 Instalar PyTorch com suporte a CUDA 12.4
- 📚 Instalar todas as dependências (ultralytics, matplotlib, etc.)
- ✅ Verificar se a GPU está disponível

#### Opção B: Instalação Manual

```bash
# Atualizar ferramentas básicas
py -m pip install --upgrade pip setuptools wheel

# Instalar PyTorch com CUDA
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Instalar demais dependências
py -m pip install -r requirements.txt
```

### 5. Verificar Instalação da GPU

Após a instalação, verifique se a GPU está sendo detectada:

```bash
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Não detectada\"}')"
```

**Saída esperada:**
```
CUDA disponível: True
GPU: NVIDIA GeForce RTX 3060
```

## 📁 Estrutura do Projeto

```
av3-yolo-training/
├── main.ipynb              # Notebook principal com todo o pipeline
├── requirements.txt        # Dependências do projeto
├── install.ps1            # Script de instalação automática
├── README.md              # Este arquivo
├── axe_dataset/           # Dataset de machados
│   ├── data.yaml          # Configuração do dataset
│   ├── train/             # Imagens e labels de treino
│   │   ├── images/
│   │   └── labels/
│   ├── valid/             # Imagens e labels de validação
│   │   ├── images/
│   │   └── labels/
│   └── test/              # Imagens e labels de teste
│       ├── images/
│       └── labels/
└── runs/                  # Resultados do treinamento (gerado automaticamente)
    └── detect/
        └── beverage_detection/
            ├── weights/   # Pesos do modelo treinado
            ├── results.png
            └── confusion_matrix.png
```

## 🚀 Como Usar

### 1. Abrir o Notebook

Abra o arquivo `main.ipynb` no VS Code ou Jupyter:

```bash
code main.ipynb
```

**No VS Code:**
- Certifique-se de selecionar o kernel correto: `venv (Python 3.13.5)`
- Clique no seletor de kernel no canto superior direito
- Escolha o interpretador em: `.\venv\Scripts\python.exe`

### 2. Executar as Células do Notebook

O notebook está organizado em seções. Execute as células na ordem:

#### **Célula 1: Importar Bibliotecas**
```python
import os, torch, matplotlib.pyplot as plt
from ultralytics import YOLO
# ... mais imports
```

✅ Verifica se a GPU está disponível

#### **Célula 2: Estatísticas do Dataset**
```python
# Mostra quantidade de imagens de treino e validação
```

📊 Exibe informações sobre o dataset

#### **Célula 3: Visualizar Amostras**
```python
visualize_sample_images(num_samples=4)
```

🖼️ Mostra amostras aleatórias com anotações

#### **Célula 4: Treinar o Modelo**
```python
model = YOLO('yolov8m.pt')
results = model.train(
    data='axe_dataset/data.yaml',
    batch=16,
    epochs=100,
    imgsz=512,
    patience=20,
    device=0
)
```

🔥 **Inicia o treinamento** (pode levar de 30 minutos a 2 horas)

**Parâmetros de Treinamento:**
- `epochs=100`: Máximo de 100 épocas
- `batch=16`: Processa 16 imagens por vez
- `imgsz=512`: Redimensiona imagens para 512x512
- `patience=20`: Para se não houver melhora em 20 épocas (early stopping)
- `device=0`: Usa a primeira GPU disponível

#### **Célula 5-8: Avaliar Resultados**

Exibe:
- Exemplos de data augmentation
- Métricas de desempenho (Precision, Recall, mAP)
- Gráficos de treinamento
- Matriz de confusão
- Predições em imagens de validação

# Importante: Apesar do modelo ter sido treinado para a detecção de bebidas, o dataset deve ser carregado em "axe_dataset/" para manter a consistência com o código fornecido.

#### **Célula 9: Testar com Imagem Customizada**
```python
test_custom_image("caminho/para/sua/imagem.jpg")
```

🎯 Testa o modelo com suas próprias imagens

### 3. Entender as Métricas

Após o treinamento, o modelo exibe métricas importantes:

| Métrica | Descrição | Valor Típico |
|---------|-----------|--------------|
| **Precision** | % de detecções corretas | 50-90% |
| **Recall** | % de objetos encontrados | 30-80% |
| **mAP@0.5** | Precisão média com IoU > 50% | 40-80% |
| **mAP@0.5:0.95** | Precisão média (IoU 50-95%) | 20-60% |

**Exemplo de output:**
```
📊 MÉTRICAS DE DESEMPENHO
============================================================
Precisão (Precision):  0.5849
Revocação (Recall):    0.3774
mAP@0.5:               0.4906
mAP@0.5:0.95:          0.2493
============================================================

💡 Interpretação:
  • Precision: De todas as detecções, 58.5% estão corretas
  • Recall: 37.7% dos machados foram encontrados
  • mAP@0.5: Precisão média com IoU > 50%
```

### 4. Testar o Modelo Treinado

Para testar com suas próprias imagens:

```python
# No notebook, execute:
test_custom_image("minha_imagem.jpg")

# Ou especifique o caminho completo:
test_custom_image(r"C:\Users\Usuario\Pictures\machado.jpg")
```

**Saída esperada:**
- 🖼️ Imagem com caixas delimitadoras verdes
- 📋 Lista de detecções com confiança
- 📍 Coordenadas das bounding boxes

## 📊 Resultados

### Arquivos Gerados

Após o treinamento, os seguintes arquivos são criados em `runs/detect/axe_detection/`:

- **`weights/best.pt`**: Melhor modelo treinado (usar para inferência)
- **`weights/last.pt`**: Último checkpoint
- **`results.png`**: Gráficos de métricas ao longo do treinamento
- **`confusion_matrix.png`**: Matriz de confusão
- **`train_batch0.jpg`**: Exemplos de data augmentation
- **`val_batch0_pred.jpg`**: Predições em imagens de validação

### Usar o Modelo Treinado em Outro Projeto

```python
from ultralytics import YOLO

# Carregar o modelo treinado
model = YOLO('runs/detect/beverage_detection/weights/best.pt')

# Fazer predição
results = model.predict('imagem.jpg', conf=0.25)

# Processar resultados
for result in results:
    boxes = result.boxes
    for box in boxes:
        print(f"Classe: {model.names[int(box.cls)]}")
        print(f"Confiança: {box.conf.item():.2%}")
        print(f"Coordenadas: {box.xyxy.tolist()}")
```

## 🐛 Troubleshooting

### GPU Não Detectada

**Problema:** `GPU disponível: False`

**Soluções:**
1. Verifique se você tem uma GPU NVIDIA:
   ```powershell
   nvidia-smi
   ```

2. Reinstale o PyTorch com CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

3. Verifique a versão do CUDA instalada:
   ```bash
   nvcc --version
   ```

### Erro: "Out of Memory" (GPU)

**Problema:** CUDA out of memory durante o treinamento

**Soluções:**
1. Reduza o batch size no treinamento:
   ```python
   model.train(batch=8)  # Em vez de 16
   ```

2. Reduza o tamanho da imagem:
   ```python
   model.train(imgsz=416)  # Em vez de 512
   ```

3. Limpe a memória da GPU antes de treinar:
   ```python
   import gc, torch
   gc.collect()
   torch.cuda.empty_cache()
   ```

### Kernel do Notebook Errado

**Problema:** Notebook usa kernel de outro projeto

**Solução:**
1. Clique no seletor de kernel (canto superior direito)
2. Selecione "Select Another Kernel..."
3. Escolha: `.\venv\Scripts\python.exe`

### Imagens Não Aparecem no Notebook

**Problema:** `<Figure size 1800x800>` mas sem imagem

**Solução:**
Adicione no início da célula:
```python
%matplotlib inline
```

### Erro ao Instalar nvidia-pyindex

**Problema:** `nvidia-pyindex` falha ao instalar

**Solução:**
Não é mais necessário! Use o método de instalação manual:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 📚 Recursos Adicionais

- [Documentação Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Roboflow - Dataset Management](https://roboflow.com/)

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Adicionar novos recursos

## 📝 Licença

Este projeto é de uso educacional.

---

**Desenvolvido com ❤️ usando YOLOv8 e PyTorch**
