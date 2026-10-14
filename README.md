# ✈️ Predição de Vida Útil Restante (RUL) de Turbofans Aeronáuticos

Projeto de Machine Learning e Deep Learning focado na predição de **RUL (Remaining Useful Life)** em motores a jato comerciais, utilizando dados de simulação de degradação da NASA (**N-CMAPSS DS01-005**) e validação em dados reais de telemetria (**NASA DASHlink**).

---

## 📌 Visão Geral da Arquitetura e Modelos

O projeto explora uma evolução progressiva de complexidade algorítmica:

```
ML/
├── Regression/                 # Baseline: Regressão Linear Simples
├── SVR/                        # Support Vector Regression + Otimização Bayesiana
├── XGBoost/                    # Gradient Boosting + Busca Bayesiana (Optuna)
├── Deep_Learning/
│   ├── CNN_1D/                 # Convolução 1D para extração de padrões de sensores
│   ├── LSTM/                   # Redes Recorrentes LSTM e Vacinação anti-overfitting
│   ├── CNN_LSTM/               # Modelo Híbrido Convolucional-Recorrente
│   ├── Suprema/                # Arquitetura profunda avançada para generalização
│   └── Transformer/            # Mecanismo de Atenção (Self-Attention) para séries temporais
├── Assets.md                   # Dicionário de sensores N-CMAPSS e guia metodológico
├── requirements.txt            # Dependências do ambiente Python
└── .gitignore                  # Regras de exclusão para datasets (>50GB) e caches
```

---

## 🔬 Modelos Implementados

### 1. Baseline Linear (`Regression/`)
- Modelo de referência estatístico ($Y = aX + b$) com Scikit-Learn.
- Avaliação com métricas $MAE$ e $RMSE$, além de gráficos de dispersão, resíduos e trajetória de degradação.

### 2. Support Vector Regression (`SVR/`)
- Treinamento com kernel RBF e amostragem inteligente para contornar a complexidade quadrática $O(n^2)$.
- Otimização Bayesiana com [Optuna](https://optuna.org/) para calibração dos hiperparâmetros $C$ e $\epsilon$.

### 3. XGBoost (`XGBoost/`)
- Árvores de decisão impulsionadas por gradiente para captura de não-linearidades complexas.
- Análise de **Feature Importance** para identificar os sensores mais críticos no desgaste do turbofan (ex: $T_{48}$, $T_{50}$, pressões e rotações).

### 4. Deep Learning (`Deep_Learning/`)
- **CNN 1D**: Extração de features espaciais e correlações locais entre canais de sensores.
- **LSTM & LSTM Vacinada**: Memória de longo prazo com regularização e injeção controlada de ruído para evitar overfitting aos dados simulados do N-CMAPSS.
- **Híbrido CNN-LSTM**: Camadas convolucionais seguidas por células LSTM, aliando extração de features e temporalidade.
- **Transformer**: Mecanismo de Self-Attention Multi-Head para dependências de longo alcance nas trajetórias dos ciclos de voo.
- **Suprema**: Rede profunda integrada para máxima robustez de generalização.

---

## 📊 Metodologia de Validação (N-CMAPSS vs. DASHlink)

1. **Feature Matching**: Alinhamento estrito dos dicionários de sensores entre o simulador N-CMAPSS e a telemetria real DASHlink (Altitude, Mach, EGT, N1, etc.).
2. **Assinatura de Saúde**: Cálculo do padrão de comportamento do modelo nos primeiros 10% da vida útil do motor (estado perfeitamente saudável).
3. **Inferência Cega no DASHlink**: Injeção contínua de voos reais de aeronaves (ex: Tail 687).
4. **Diagnóstico Overfitting vs. Generalização**:
   - **Boa Generalização**: RUL estimada alta e estável ao longo de voos normais.
   - **Overfitting**: RUL oscilando drasticamente com mudanças de altitude/empuxo ou reportando falha iminente.

---

## 🚀 Como Executar

### Pré-requisitos
Instale as dependências com Python 3.9+:

```bash
pip install -r requirements.txt
```

### Executando os Treinamentos

- **Baseline Linear**:
  ```bash
  python Regression/Baseline_Linear.py
  ```

- **SVR com Otimização Bayesiana**:
  ```bash
  python SVR/Bayesiana.py
  ```

- **XGBoost**:
  ```bash
  python XGBoost/Train_XGBoost.py
  ```

- **Deep Learning (LSTM / CNN / Transformer)**:
  ```bash
  python Deep_Learning/LSTM/Train_LSTM.py
  python Deep_Learning/Transformer/Train_Transformer.py
  ```

> ⚠️ **Nota sobre os Datasets**: Os arquivos brutos `.h5` da NASA N-CMAPSS e os zips do DASHlink somam mais de 50 GB e estão devidamente ignorados no `.gitignore`. Baixe-os diretamente dos repositórios oficiais da [NASA](https://data.nasa.gov/) e aponte os caminhos configurados nas variáveis `caminho_h5` dos scripts.
