# Diário de Pesquisa: CNN 1D (Rede Neural Convolucional Unidimensional)

## 1. Fundamentação Matemática e Funcionamento Prático
A filosofia por trás de aplicar uma CNN 1D em séries temporais (em contraste com o ciclo recursivo infinito da LSTM) baseia-se num princípio chamado de **Invariância Translacional Local**. A CNN tenta encontrar "Padrões Clássicos" (features) que significam uma coisa, independentemente de quando aconteceram. Ela faz isso escorregando filtros matemáticos.

**A Matemática da Convolução 1D:**
$$y(t) = (x * k)(t) = \sum_{\tau} x(t - \tau) k(\tau)$$
Sendo $x$ o sinal cru dos 45 sensores e $k$ o "Filtro" (Kernel) criado na Rede Neural.

*Exemplo Prático:*
Imagine que uma turbina com defeito na câmara sempre apresente um surto exato: Queda de pressão seguida imediatamente de um choque brusco positivo `(Baixo, Alto)`. 
A Rede CNN treina um kernel interno de pesos idêntico: `[-1.0, +1.0]`. 
Quando a CNN desliza essa pequena matriz `k` pelo tempo e encontra um momento calmo `[100, 101]`, a multiplicação retorna zero (Ruído passivo nulo). Porém, ao cruzar pelo choque fatal da turbina de `[100, 150]`, a multiplicação das matrizes reage explosivamente gerando uma flag altíssima. Esse é o **Feature Map** que será lido à frente. Em suma: A rede usa scanner de assinaturas invés de ler um diário.

As ativações nulas são silenciadas pela matemática **ReLU** ($f(x) = \max(0, x)$), enquanto a camada de **MaxPool1d** comprime os dados ignorando lixo e repassando apenas o pico máximo do Feature Map adiante.

## 2. Diário de Falhas, Gargalos e Soluções (A Física dos Tensores)
| Problema / Falha Crítica | Causa Raiz | Ação Corretiva Exata |
| :--- | :--- | :--- |
| **Incompatibilidade Geométrica de Tensores na Arquitetura (Erro `Runtime`)** | O PyTorch desenhou nativamente a camada `nn.Conv1d` não para tempo, mas para canais. Se você entregar o dado no formato `[Batch, Tempo, Sensores]`, ela vai cruzar os núcleos matemáticos na direção do tempo invés de cruzar nos sensores, o que geraria um Feature Map falso. | Programamos uma disrupção em baixo nível diretamente no estágio `forward()` do código de Treinamento: `x = x.transpose(1, 2)`. Giramos as dimensões matemáticas de 1 e 2 para que os kernels viajassem nas trilhas certas `[Batch, Canais(Sensores), Tempo]`. |
| **Esgotamento de Memória Global** | Reproduzir o processamento cronológico de janelas deslizantes sem explodir buffers internos limitados em Windows Systems. | Mantida a herança defensiva perfeita originária da scriptagem da LSTM: `device='cpu'` e Scaling Matemático Modificado *In-Place*. |

## 3. Arquitetura e Hiperparâmetros Finais
- **Camada Convolucional 1:** Lendo 64 filtros de núcleo de comprimento temporal (kernel) 3 (enxerga 3 voos simultâneos cruzando todos os 45 sensores), Padding 1. Adição da ReLU matemática acoplada de MaxPool1d(2), caindo e achatando a janela crônica temporal bruscamente pela metade (30 -> 15 passos).
- **Camada Convolucional 2:** Salto de expansão para 128 filtros de Kernel 3 com MaxPool1d(2). Corta e condensa a janela final para meros 7 passos ultra-ricos de contexto temporal.
- **Flatten (Achatamento Dimensional):** Quebra do tensor de matriz dimensional para um super-array linear clássico composto pelas multiplicações (128 canais * 7 passos restantes = 896 nós cruciais).
- **Rede Densa Total (FC):** 64 neurônios com um bloqueio restrito via Dropout de pesados 30% ($0.3$), desaguando linearmente no número alvo de RUL (Ciclos).

## 4. Evolução do Treinamento e Resultado Quantitativo
Isolada em CPU (`device='cpu'`), a CNN iterou os laços validando em cross-set com as métricas gravadas no tempo de 96.86 minutos totais.

| Época | RMSE Treino (Ciclos) | RMSE Teste Cego (Ciclos) | Análise Comportamental |
| :--- | :--- | :--- | :--- |
| 1 | 6.86 | 7.36 | Filtros aleatórios buscando vetores coesos. |
| 2 | 5.95 | 7.23 | Refino direcional dos Kernels. |
| **3** | 5.65 | **6.96** | **(Ponto Ótimo)** Padrões geométricos encontrados funcionam com boa simetria para turbinas estrangeiras. |
| 5 | 5.16 | 6.99 | Filtros começam a engessar e rejeitar anomalias cegas. |
| 7 | 4.99 | 7.84 | Estagnação e desconexão entre domínios. |
| **10** | **4.78** | **8.11** | Decoreba geométrica final. |

### Conclusão Analítica
Quando comparada frontalmente à volatilidade da arquitetura LSTM (que foi para incríveis 1.78), percebemos categoricamente por prova física que as Convolucionais são restritivas e engessadas (terminando o laço em severos 4.78). Por procurar padrões curtos em vez de reescrever matrizes de fita transportadora de dados completos, a CNN 1D sofre uma quantidade consideravelmente mais leve e suportável de "Overfitting Catastrófico". No entanto, sem a capacidade lógica estrutural para medir conexões de longuíssimo tempo sistêmicas, seu poder preditivo não é capaz de desvendar a variância final cega presente nos sets da base CMAPSS DS02, perdendo irremediavelmente para as Random Forests baseadas em hist gradient boosting (XGBoost).
