# Ovary Image Classification

Experimentos comparativos de arquiteturas CNN para classificação de imagens de ultrassom ovariano. Trabalho da disciplina **Aplicação de Sistemas Inteligentes**.

Dataset: 3 classes — `dominant_follicle`, `normal`, `PCO` — em ultrassom ovariano.

## Estrutura

```
.
├── main.py                              # Split treino/teste + data augmentation (albumentations)
├── requirements.txt                     # Dependências mínimas (PyTorch + torchvision + sklearn)
├── membro1_efficientnet.ipynb           # 8 variações: EfficientNet B2 → B3
├── membro2_resnet_resnext.ipynb         # 8 variações: ResNet-50/101 + ResNeXt-50
├── membro3_densenet.ipynb               # 8 variações: DenseNet-121 → 169
└── membro4_convnext_mobilenet (1).ipynb # 8 variações: ConvNeXt-T/S + MobileNetV3-L
```

Cada notebook segue o `Plano_Experimentos_CNN.docx` e implementa 8 variações escalonadas — 2 Muito Leves, 2 Leves, 2 Médias e 2 Pesadas — com `train_model` + early stopping, e CutMix / RandAugment / TTA nas variações pesadas.

## Distribuição de arquiteturas

| Membro | Notebook | Backbone | Ponto forte |
|--------|----------|----------|-------------|
| 1 | `membro1_efficientnet.ipynb` | EfficientNet B2 → B3 | Eficiência parâmetro/desempenho |
| 2 | `membro2_resnet_resnext.ipynb` | ResNet-50/101 + ResNeXt-50 | Baseline clássico robusto |
| 3 | `membro3_densenet.ipynb` | DenseNet-121 → 169 | Reutilização de features (datasets pequenos) |
| 4 | `membro4_convnext_mobilenet (1).ipynb` | ConvNeXt-T/S + MobileNetV3-L | Modelo leve vs. moderno |

## Setup

```bash
# 1. Clonar o repo
git clone https://github.com/Wakian/ovary-image-classification.git
cd ovary-image-classification

# 2. Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux/macOS

# 3. Instalar dependências
pip install -r requirements.txt
pip install jupyter timm matplotlib albumentations opencv-python joblib
```

## Dataset

O dataset **não está versionado** (fica fora do repo por causa do tamanho). Estrutura esperada:

```
../data_sets/ovarian_ultrasound_dataset/
├── dominant_follicle/
├── normal/
└── PCO/
```

Cada notebook detecta automaticamente se há split `train/validation` pré-definido; caso contrário faz split estratificado 80/20.

## Pipeline de execução

**1. Augmentation (uma vez, fora dos notebooks):**

```bash
python main.py
```

Faz split treino/teste **antes** da augmentation (garante zero vazamento) e gera 1000 imagens por classe no diretório `OUTPUT_PATH` (ajustar no topo de `main.py`).

**2. Treino (cada membro roda seu notebook):**

Cada variação V1–V8 produz:
- Accuracy / F1 Macro / Precision / Recall / Loss / Tempo (min)
- Curvas treino vs validação
- Matriz de confusão normalizada
- Modelo salvo em `modelos_salvos/membro{N}/`

**3. Ensemble final:**

Selecionar os 3–5 melhores resultados do grupo e combinar via média ponderada das probabilidades.

## Métricas a registrar (obrigatórias)

Para cada uma das 32 variações (8 por membro):

| Var. | Peso | Acc | F1 Macro | Precision | Recall | Tempo (min) | Observações |
|------|------|-----|----------|-----------|--------|-------------|-------------|
| V1   | Muito Leve | — | — | — | — | — | — |
| ...  | ... | ... | ... | ... | ... | ... | ... |

Templates prontos no final de cada notebook.

## Notas

- Notebooks usam `epochs=1000` com `patience=25` de early stopping nas variações V1–V6; V7/V8 usam o número fixo de épocas do plano (cosine annealing precisa de horizonte definido).
- `modelos_salvos/` (~640 MB com pesos completos) é ignorado pelo Git — modelos ficam locais.
