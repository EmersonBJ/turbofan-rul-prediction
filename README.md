# Predicao de Vida Util Restante (RUL) de Turbofans Aeronauticos

Estudo comparativo de modelos de Machine Learning e Deep Learning aplicados a estimativa da Vida Util Restante (Remaining Useful Life - RUL) de motores a jato comerciais, utilizando o dataset de degradacao simulada NASA N-CMAPSS (DS01-005 / DS02) e protocolos de validacao para telemetria real (NASA DASHlink).

---

## 1. Escopo e Objetivos Tecnicos

O objetivo deste projeto consiste em avaliar rigorosamente diferentes paradigmas de modelagem estatistica e conexionista frente a dados de sensores de turbomotores sob condicoes severas de voo (altitude, Mach, angulo da manete e temperaturas de exaustao).

A analise foca em tres questoes centrais da engenharia preditiva:
1. Capacidade de generalizacao em cenarios cegos de teste contra o fenonemo de overfitting decorrente do ruido dos dados simulados.
2. Trade-off entre complexidade computacional (tempo de convergencia, uso de memoria) e precisao metrica (RMSE e MAE).
3. Eficacia de abordagens baseadas em arvores de decisao (Gradient Boosting) versus arquiteturas neurais recorrentes e convolucionais (Deep Learning).

---

## 2. Estrutura do Repositorio

```
ML/
|-- Assets.md                             # Dicionario de variaveis N-CMAPSS e alinhamento de sensores
|-- requirements.txt                      # Dependencias do ambiente Python
|-- .gitignore                            # Politicas de exclusao de dados brutos e caches
|-- Regression/
|   |-- Baseline_Linear.py                # Modelo linear de referencia (Scikit-Learn)
|   |-- modelo_baseline_linear.pkl        # Artefato serializado do baseline linear
|   |-- tabela_erros_linear.csv           # Metricas registradas do modelo linear
|   |-- grafico_1_trajetoria_linear.png   # Curva de degradacao estimada
|   |-- grafico_2_dispersao_linear.png    # Dispersao Real vs Predito
|   `-- grafico_3_residuos_linear.png     # Distribuicao residual dos erros
|-- SVR/
|   |-- Bayesiana.py                      # Otimizacao Bayesiana de hiperparametros (Optuna)
|   |-- Train.py                          # Treinamento com kernel RBF e subamostragem
|   |-- Test.py                           # Rotina de teste e validacao cruzada
|   |-- modelo_rul_treinado.pkl           # Artefato treinado do SVR
|   `-- Models/                           # Pesos intermediarios de variacoes de C e Epsilon
|-- XGBoost/
|   |-- Train_XGBoost.py                  # Treinamento otimizado com histogram-based method
|   |-- Bayesiana_XGBoost.py              # Busca bayesiana de taxa de aprendizado e profundidade
|   |-- XGBoost_Explicacao.md             # Fundamentacao teorica da expansao de Taylor no boosting
|   |-- modelo_xgboost_otimizado.pkl      # Modelo campeao serializado (melhor performance geral)
|   |-- tabela_erros_xgboost.csv          # Metricas de erro do XGBoost
|   |-- grafico_1_trajetoria_xgboost.png  # Trajetoria preditiva do XGBoost
|   `-- grafico_2_feature_importance.png  # Relevancia relativa de cada sensor
`-- Deep_Learning/
    |-- CNN_1D/
    |   |-- Train_CNN.py                  # Extracao de padroes espaciais via convolucao 1D
    |   |-- modelo_cnn.pth                # Pesos treinados da CNN 1D
    |   |-- Diario_CNN.md                 # Registro tecnico e curva de aprendizado
    |   |-- grafico_cnn_learning_curve.png
    |   `-- grafico_cnn_trajetoria.png
    |-- LSTM/
    |   |-- Train_LSTM.py                 # Rede Recorrente LSTM (Long Short-Term Memory)
    |   |-- Train_LSTM_Vacinada.py        # Injecao de ruido para controle de overfitting
    |   |-- Exportar_Netron.py            # Serializacao e exportacao da malha para formato ONNX
    |   |-- RaioX_LSTM.onnx               # Grafo neural ONNX
    |   |-- modelo_lstm.pth               # Pesos treinados da LSTM
    |   `-- Diario_LSTM.md                # Analise matematica de dissipacao do gradiente
    |-- CNN_LSTM/
    |   |-- Train_CNN_LSTM.py             # Arquitetura hibrida convolucional-recorrente
    |   |-- modelo_hibrido.pth            # Pesos treinados do modelo hibrido
    |   `-- Diario_Hibrida.md             # Analise de colapso por duplicidade de gradientes
    |-- Suprema/
    |   |-- Train_Suprema.py              # Bi-LSTM com Huber Loss, AdamW e taxa adaptativa
    |   |-- modelo_supremo.pth            # Pesos treinados do modelo supremo
    |   `-- Diario_Suprema.md             # Analise da convergencia simetrica temporal
    `-- Transformer/
        |-- Train_Transformer.py          # Encoder de Self-Attention Multi-Head
        |-- modelo_transformer.pth        # Pesos treinados do Transformer
        `-- Diario_Transformer.md         # Analise de feature engineering com medias moveis
```

---

## 3. Ranking de Desempenho dos Modelos

A tabela abaixo sumariza o ranking oficial de performance dos algoritmos testados na base cega de validacao, ordenados do melhor para o pior desempenho (menor RMSE):

| Posicao | Modelo / Arquitetura | RMSE Teste (Ciclos) | MAE (Ciclos) | Complexidade / Tempo | Artefato / Arquivo |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1º** | **XGBoost Otimizado (Hist Gradient Boosting)** | **4.48** | **4.53** | ~150 s (CPU) | `XGBoost/modelo_xgboost_otimizado.pkl` |
| **2º** | **Rede Suprema (Bi-LSTM + Huber Loss + AdamW)** | **6.39** | -- | ~160 min (CPU) | `Deep_Learning/Suprema/modelo_supremo.pth` |
| **3º** | **LSTM Padrao (Long Short-Term Memory)** | **6.48** | -- | ~170 min (CPU) | `Deep_Learning/LSTM/modelo_lstm.pth` |
| **4º** | **Hibrido CNN-LSTM (Extrator Espacial + Recorrencia)** | **6.83** | -- | ~138 min (CPU) | `Deep_Learning/CNN_LSTM/modelo_hibrido.pth` |
| **5º** | **CNN 1D (Convolucao Unidimensional)** | **6.96** | -- | ~96 min (CPU) | `Deep_Learning/CNN_1D/modelo_cnn.pth` |
| **6º** | **SVR (Support Vector Regression RBF + Optuna)** | **7.09** | -- | Amostrado (O(n^2)) | `SVR/modelo_rul_treinado.pkl` |
| **7º** | **Transformer Encoder (Mecanismo de Auto-Atencao)** | **7.62** | -- | ~120 min (CPU) | `Deep_Learning/Transformer/modelo_transformer.pth` |
| **8º** | **Regressao Linear Simples (Baseline Estatistico)** | **15.97** | **14.68** | Instantaneo | `Regression/modelo_baseline_linear.pkl` |

---

## 4. Analise Tecnica Individual dos Modelos

### 1º Lugar: XGBoost Otimizado (Melhor Modelo Geral)
- **RMSE:** 4.48 ciclos | **MAE:** 4.53 ciclos.
- **Configuracao:** `learning_rate=0.1955`, `max_depth=3`, `subsample=0.8888`, `colsample_bytree=0.7060`, `min_child_weight=8`, `tree_method='hist'`.
- **Fatores de Sucesso:**
  - O metodo baseado em histogramas converteu atributos continuos de 45 colunas em bins discretos, reduzindo a complexidade de operacao de $O(N)$ para $O(\text{bins})$.
  - A regularizacao nativa com penalidade sobre folhas e normas dos pesos ($\Omega(f_t) = \gamma T + \frac{1}{2}\lambda ||w||^2$) conferiu imunidade superior aos transientes de voo e picos espurios de telemetria.
  - A profundidade reduzida (`max_depth=3`) preveniu ativamente a decoracao de correlacoes espurias de sensores.

### 2º Lugar: Rede Suprema Bi-LSTM (Melhor Modelo de Deep Learning)
- **RMSE:** 6.39 ciclos na Epoca 2 (com intervencao do Early Stopping).
- **Configuracao:** Camada LSTM bidirecional (128 nos concatenados) processando janelas temporais de 30 ciclos em direcao direta e reversa ($x_1 \dots x_{30}$ e $x_{30} \dots x_1$).
- **Fatores de Sucesso:**
  - Emprego de **Smooth L1 Loss (Huber Loss)**, alternando comportamento quadratico para deltas pequenos e linear para desvios acentuados, neutralizando explosoes de gradiente causadas por leituras anomalas de temperatura.
  - Otimizador **AdamW** com desacoplamento rigoroso do decaimento de pesos (`weight_decay=1e-4`).
  - **Gradient Clipping** (`max_norm=1.0`) associado a escalonador de taxa adaptativa (`ReduceLROnPlateau`).

### 3º Lugar: LSTM Padrao
- **RMSE:** 6.48 ciclos na Epoca 2 (ponto otimo); deterioracao para 8.04 ciclos na Epoca 10 sem Early Stopping.
- **Configuracao:** 2 camadas empilhadas com 64 celulas por camada e Dropout de 0.2.
- **Limitacoes:**
  - O Estado Celular ($C_t$) demonstrou sensibilidade pronunciada a ruidos estocasticos no conjunto de treino, resultando em sobreajuste critico (RMSE de treino convergiu para 1.78, enquanto o erro de teste divergiu).

### 4º Lugar: Hibrido CNN-LSTM
- **RMSE:** 6.83 ciclos na Epoca 1; convergencia para 7.62 ciclos na Epoca 10.
- **Configuracao:** Camada convolucional 1D com 64 filtros de tamanho 3 seguida por MaxPool1d(2), alimentando 2 camadas de celulas LSTM de 64 nos.
- **Limitacoes:**
  - O modelo apresentou ganho de convergencia expressivo no estagio inicial. Todavia, a densidade excessiva de parametros livres acelerou a memorizacao das trajetorias de treino (RMSE de treino atingiu 1.53), demandando mecanismos mais agressivos de regularizacao.

### 5º Lugar: CNN 1D Pura
- **RMSE:** 6.96 ciclos na Epoca 3; saturacao em 8.11 ciclos na Epoca 10.
- **Configuracao:** Dois estagios convolucionais com 64 e 128 canais, ativacao ReLU, MaxPool1d(2) e achatamento dimensional para 896 nos lineares com Dropout de 0.3.
- **Limitacoes:**
  - A hipotese de invariancia translacional local foi eficaz para correlacoes simultaneas de sensores, mas a ausencia de recorrencia temporal limitou a modelagem da degradacao acumulada ao longo do ciclo de vida util.

### 6º Lugar: Support Vector Regression (SVR)
- **RMSE:** 7.09 ciclos.
- **Configuracao:** Kernel de Base Radial (RBF) com ajuste bayesiano dos coeficientes $C$ e $\epsilon$.
- **Limitacoes:**
  - Custo computacional assintotico quadratico no calculo da matriz de distancias ponto a ponto ($O(N^2)$), inviabilizando o treinamento sobre o volume integral de 5 milhoes de registros e demandando amostragem reduzida (20.000 linhas).

### 7º Lugar: Transformer Encoder
- **RMSE:** 7.62 ciclos.
- **Configuracao:** Camada de Transformer Encoder (PyTorch `nn.TransformerEncoderLayer`) com mecanismo de Multi-Head Self-Attention, Positional Encoding e pre-processamento por media movel (`scipy.ndimage.uniform_filter1d`).
- **Limitacoes:**
  - A auto-atencao requer volumes expressivos de historico continuo e foi excessivamente afetada pela variabilidade intermitente dos ciclos de voo.

### 8º Lugar: Regressao Linear (Baseline)
- **RMSE:** 15.97 ciclos | **MAE:** 14.68 ciclos.
- **Limitacoes:**
  - Natureza unicamente linear ($Y = \sum w_i x_i + b$), incapaz de descrever as descontinuidades de desgaste, saturacao termica de turbinas de alta pressao e efeitos transientes de operacao.

---

## 5. Engenharia de Dados e Restricoes de Hardware

O processamento do dataset N-CMAPSS DS01/DS02 (~5 milhoes de registros x 45 variaveis fisicas) impos desafios criticos de infraestrutura:

1. **Gestao de Memoria:**
   - Evitou-se o consumo redundante de memoria via clonagem de DataFrames do Pandas, adotando manipulacao direta de matrizes `numpy.float32` extraidas sob demanda do formato HDF5 (`h5py`).
   - Normalizacao executada atraves de operacoes matematicas in-place para mitigar picos de alocacao dinamica.
2. **Carregamento Lazy de Sequencias 3D:**
   - Em vez de gerar janelas tridimensionais estaticas de 30 voos (o que demandaria alocacoes superiores a 25 GB na RAM), foi implementada uma subclasse customizada de `torch.utils.data.Dataset` com fatiamento pontual no metodo `__getitem__`.
3. **Estabilidade de Execucao:**
   - Assegurou-se a operacao deterministica em processador central (CPU), evitando falhas de alocacao em VRAM para modelos de multiplas camadas recorrentes.

---

## 6. Conclusao Cientifica

Os experimentos confirmam uma conclusao empirica consistente com a literatura de engenharia de confiabilidade aeronautica:
- Para dados tabulares de telemetria industrial marcados por ruidos estocasticos de operacao atmosferica, **algoritmos de ensemble baseados em arvores de decisao com regularizacao de gradiente (XGBoost) superam sistematicamente arquiteturas de Deep Learning convencionais**.
- O modelo **XGBoost Otimizado** atingiu a lideranca absoluta com **4.48 ciclos de RMSE**, contra **6.39 ciclos** alcancados pela arquitetura neural mais sofisticada (**Rede Suprema Bi-LSTM**), consumindo uma fracao infima do custo computacional de treinamento (~150 segundos vs ~160 minutos).

---

## 7. Instrucoes de Reproducao

### Instalacao do Ambiente
```bash
pip install -r requirements.txt
```

### Execucao dos Treinamentos e Avaliacoes

- Treinamento do Modelo Campeao (XGBoost):
  ```bash
  python XGBoost/Train_XGBoost.py
  ```

- Treinamento da Rede Suprema (Bi-LSTM):
  ```bash
  python Deep_Learning/Suprema/Train_Suprema.py
  ```

- Treinamento do Baseline Linear:
  ```bash
  python Regression/Baseline_Linear.py
  ```

- Inferencia e Teste com SVR:
  ```bash
  python SVR/Test.py
  ```
