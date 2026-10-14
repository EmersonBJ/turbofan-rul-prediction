# Diário de Pesquisa: LSTM (Long Short-Term Memory)

## 1. Fundamentação Matemática e Funcionamento Prático
A LSTM é desenhada para combater o problema da "dissipação do gradiente" (Vanishing Gradient) das Redes Neurais Recorrentes clássicas. Ela não apenas enxerga o tempo atual, mas mantém um **Estado Celular ($C_t$)** — uma fita transportadora de memória de longo prazo controlada por três portas (Gates) matemáticas:

1. **Forget Gate (Porta de Esquecimento - $f_t$):** Decide o que apagar da memória baseada no sinal atual $x_t$ e memória anterior $h_{t-1}$.
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
   *Exemplo Prático:* Se a turbina muda de regime de altitude abruptamente, a LSTM ativa essa porta próxima de `0` para "esquecer" o baseline de pressão da altitude antiga, impedindo falhas de interpretação.
2. **Input Gate (Porta de Entrada - $i_t$):** Decide que novas informações salvar.
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
   *Exemplo Prático:* A rede detecta um leve aumento constante de temperatura (um novo sintoma). O Input Gate valida e imprime essa tendência na fita celular $C_t$.
3. **Atualização do Estado ($C_t$):**
   $$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$
4. **Output Gate (Porta de Saída - $o_t$):** Decide qual parte da memória será jogada para fora (a Previsão).
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t * \tanh(C_t)$$

## 2. Diário de Falhas, Gargalos e Soluções (Engenharia de Dados)
O dataset N-CMAPSS DS02 possui quase 5 milhões de linhas e 45 colunas (`A`, `T`, `W`, `Xs`, `Xv`), o que nos levou aos limites absolutos de hardware da máquina de desenvolvimento:

| Problema / Falha Crítica | Causa Raiz | Ação Corretiva Exata |
| :--- | :--- | :--- |
| **`ArrayMemoryError` ao carregar Pandas** | O uso de DataFrames clonou metadados desnecessários da base de 4.9M de linhas. | Puxamos tensores brutos `numpy.float32` via pacote `h5py`, mapeando os vetores da linguagem C diretamente para a memória. |
| **Estouro de RAM de +842MB no Scaling** | O `MinMaxScaler` nativo do Scikit-Learn instanciou uma matriz "cópia espelho" oculta na RAM para calcular o escore. | Omitimos a biblioteca e codificamos a normalização com operações matemáticas em baixo nível (Memória *In-Place*): `X_treino -= min_vals` e `X_treino /= range_vals`. Os bytes originais foram sobrescritos na própria RAM para impedir pico de memória. |
| **OOM (Out-Of-Memory) no fatiamento 3D** | Para que a LSTM lesse 30 voos retroativos a cada passo temporal, alocar todos os 4.9M de linhas multiplicados por 30 criaria um arquivo invisível de mais de 25 Gigabytes. | Criamos uma Classe PyTorch `CMAPSS_SequenceDataset` com **Lazy Loading**. Mapeamos ponteiros de índices, e a fatia `[Batch, 30, 45]` só é extraída cirurgicamente do HD no `__getitem__` (um milissegundo antes da LSTM mastigar), evitando alocação estática. |
| **`CUDA Memory allocation failure`** | Ao disparar o `loss.backward()`, os Tensores Gradientes retroativos da Placa de Vídeo (Backpropagation de dados 3D simultâneos) não couberam na VRAM. | Forçamos a variável `device = torch.device('cpu')`, desviando o carregamento da Placa de Vídeo para aproveitar a enorme Memória Virtual do Processador central, reduzindo também o `batch_size` de 256 para `64`. |

## 3. Arquitetura e Hiperparâmetros Finais
- **Entrada (Input):** 45 Sensores.
- **Janela de Tempo (Sequence Length):** 30.
- **Camadas Escondidas (Hidden Size):** 64 neurônios, 2 camadas empilhadas (num_layers=2).
- **Regularização Básica:** Dropout de 0.2 (20% de inibição neuronal).
- **Camada Densa (Linear):** 64 para 1 nó solitário final (Saída Regressiva da RUL).

## 4. Evolução do Treinamento e Resultado Quantitativo
Com a métrica de validação cega ativada a cada época (para plotarmos a curva verdadeira), a LSTM demorou 170.19 minutos processando em CPU. Eis os valores crus recuperados dos logs:

| Época | RMSE Treino (Ciclos) | RMSE Teste Cego (Ciclos) | Análise Comportamental |
| :--- | :--- | :--- | :--- |
| 1 | 6.77 | 7.66 | A rede encontra correlações generalistas grosseiras nas turbinas. |
| **2** | 2.91 | **6.48** | **(Ponto Ótimo Absoluto)** A LSTM compreendeu regras universais de degradação aplicáveis a qualquer motor NASA. |
| 3 | 3.26 | 7.81 | Divergência inicia bruscamente. O motor memoriza ruído e erra no teste de estresse novo. |
| 4 | 2.57 | 7.49 | O abismo entre o treino e a realidade aumenta. |
| 7 | 1.98 | 7.47 | |
| **10** | **1.78** | **8.04** | A rede falhou. Dominou a simulação e errou no mundo real. |

### Conclusão Analítica
A densa matemática das "Portas de Memória" da LSTM foi a sua própria ruína perante um dataset brutal e variante como o N-CMAPSS DS02. Em vez de abstrair tendências sistêmicas como faria num texto ou áudio puro, o algoritmo aproveitou-se de sua incrível capacidade de armazenar informações ($C_t$) para mapear assinaturas mecânicas microscópicas exclusivas dos motores do conjunto de Treinamento (onde sua pontuação de Erro caiu ao número genial de 1.78 ciclos).

Resultado: Num clássico processo visível no gráfico de **Overfitting Comprovado**, a pontuação no Teste afundou em 8.04. Na ausência de recursos programados como *Early Stopping* programático ativado no código logo após a Época 2 (onde o pico reverso aconteceu), a arquitetura inteira empalidece em comparação matemática ao Boosting Gradient tabular cru e não modificado (XGBoost com 4.48 de RMSE).
