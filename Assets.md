https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/



Alinhamento de Variáveis (Feature Matching)
Garantindo que o modelo fale a mesma língua
Antes de escrever qualquer código de ML, você precisa cruzar os dicionários de dados do N-CMAPSS e do DASHlink. Isole estritamente as variáveis presentes em ambos os conjuntos (ex: Altitude, Mach, Temperatura de Exaustão - EGT, Rotação do Fan - N1). Aplique a mesma técnica de normalização (como Min-Max Scaler) nos dois datasets, usando os parâmetros extraídos apenas do N-CMAPSS.

2
Treinamento e Validação no N-CMAPSS
Aprender a física da degradação
Treine seu algoritmo (SVM, Random Forest ou Rede Neural) usando os dados do N-CMAPSS. Separe o N-CMAPSS no formato clássico (70% treino, 30% teste) para garantir que o modelo consegue prever a Vida Útil Restante (RUL) com baixo erro (RMSE) dentro do ambiente controlado.

3
Definição da Assinatura de Saúde
Estabelecendo o comportamento base
Nos dados do N-CMAPSS, isole as predições feitas nos primeiros 10% da vida útil de cada motor. Essa é a "assinatura" de como o seu modelo reage quando sabe que o motor está perfeitamente saudável. Calcule a média e a variância da RUL prevista nessa fase.

4
Inferência Cega no DASHlink
O teste no mundo real
Pegue milhares de voos de uma aeronave específica do DASHlink (como o Tail 687) e injete no seu modelo treinado. Peça para o algoritmo prever a RUL contínua ao longo de todos esses voos reais.

5
Diagnóstico de Overfitting vs. Generalização
Análise de estabilidade da predição
Plote as predições feitas no DASHlink e compare com a "assinatura saudável" do Passo 3.

Se a RUL prevista no DASHlink for alta e formar uma linha de tendência estável (com ruído aceitável): Boa Generalização.

Se a RUL oscilar violentamente (ex: caindo para zero na subida e subindo na descida) ou prever falha iminente o tempo todo: Overfitting severo. O modelo decorou os parâmetros do simulador e quebrou ao ver o ruído real.

---

# Dicionário de Variáveis do N-CMAPSS (Guia Completo)

Este guia mapeia o significado de cada coluna dentro das matrizes (`keys`) do arquivo `.h5`, combinando a documentação da NASA com os arrays `_var`.

### 1. Condições de Voo / Cenário (`W_dev` e `W_test`)
*Aeronáutica básica: dita em qual ambiente o motor está operando no momento da leitura.*
- **`alt`**: Altitude de voo (pés - ft).
- **`Mach`**: Número de Mach (Velocidade em relação ao som).
- **`TRA`**: Throttle Resolver Angle (%) - O ângulo da manete de potência (acelerador).
- **`T2`**: Temperatura Total na entrada do Fan (°R).

### 2. Sensores Físicos (`X_s_dev` e `X_s_test`)
*Sensores que realmente existem em um avião comercial e podem ser medidos em voo.*
- **`T24`**: Temperatura Total na saída do LPC (°R).
- **`T30`**: Temperatura Total na saída do HPC (°R).
- **`T48`**: Temperatura Total na saída da HPT (°R).
- **`T50`**: Temperatura Total na saída da LPT (°R).
- **`P15`**: Pressão Total no Bypass Duct (psia).
- **`P2`**: Pressão Total na entrada do Fan (psia).
- **`P21`**: Pressão Total na saída do Fan (psia).
- **`P24`**: Pressão Total na saída do LPC (psia).
- **`Ps30`**: Pressão Estática na saída do HPC (psia).
- **`P40`**: Pressão Total na saída do Combustor (psia).
- **`P50`**: Pressão Total na saída da LPT (psia).
- **`Nf`**: Rotação Física do Fan (RPM).
- **`Nc`**: Rotação Física do Core (RPM).
- **`Wf`**: Fluxo de Combustível (lbs/sec).

### 3. Sensores Virtuais (`X_v_dev` e `X_v_test`)
*Métricas calculadas termodinamicamente que não possuem sensores físicos na vida real.*
- **`T40`**: Temperatura na saída do Combustor (°R).
- **`P30`**: Pressão Total na saída do HPC (psia).
- **`P45`**: Pressão Total na saída da HPT (psia).
- **`W21`**: Fluxo de ar no Fan (lbs/sec).
- **`W22`**: Fluxo de ar no Core (lbs/sec).
- **`W25`, `W31`, `W32`**: Fluxos de sangria (bleed) de ar para refrigeração.
- **`W48`, `W50`**: Fluxos de exaustão das turbinas.
- **`SmFan`, `SmLPC`, `SmHPC`**: Margem de Estol (Stall margin) dos compressores.
- **`phi`**: Razão de combustível para a pressão Ps30.

### 4. Parâmetros Reais de Saúde (`T_dev` e `T_test`)
*Os modificadores invisíveis ($\theta$) que o simulador altera lentamente para imitar o desgaste da máquina.*
- **`fan_eff_mod` / `fan_flow_mod`**: Perda de eficiência e fluxo no Fan.
- **`LPC_eff_mod` / `LPC_flow_mod`**: Perda de eficiência e fluxo no LPC.
- **`HPC_eff_mod` / `HPC_flow_mod`**: Perda de eficiência e fluxo no HPC.
- **`HPT_eff_mod` / `HPT_flow_mod`**: Perda de eficiência e fluxo na HPT.
- **`LPT_eff_mod` / `LPT_flow_mod`**: Perda de eficiência e fluxo na LPT.

### 5. Metadados Auxiliares (`A_dev` e `A_test`)
*Identificação espacial e temporal de cada linha de dados.*
- **`Unit_ID`**: ID numérico do motor (Ex: Aeronave 1, 2, 3...).
- **`Cycle`**: Número do voo atual (começa em 1 e vai até a quebra).
- **`Fc`**: Classe de Voo (1=Curto, 2=Médio, 3=Longo).
- **`hs`**: Gabarito do Estado de Saúde (0 = Saudável, 1 = Falhando).

### 6. Alvo Predicional (`Y_dev` e `Y_test`)
*O que o algoritmo de Machine Learning precisa aprender a prever.*
- **`RUL`**: Remaining Useful Life (Vida Útil Restante em ciclos/voos).


Passo 0
Montagem de vector A para uma aeronave A

A = ['id', 'cycle';'alt', 'Mach', 'TRA', 'T2'; 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf'; 'T40', 'P30', 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi']

Matriz de entrada A id A cada linha representa uma caracteristica/aba do h5 cada coluna representa uma observação/voo para um tempo = cycle


para tanto devemos concatenar(colar) os arquivos dev para treinar a maquina

A = arquivo_h5['A_dev']
varA = arquivo_h5['A_var']

T = arquivo_h5['T_dev']
varT = arquivo_h5['T_var']

W = arquivo_h5['W_dev']
varW = arquivo_h5['W_var']

Xs = arquivo_h5['X_s_dev']
varXs = arquivo_h5['x_s_var']

Xv = arquivo_h5['X_v_dev']
varXv = arquivo_h5['X_v_var']

Y = arquivo_h5['Y_dev']
varY = arquivo_h5['Y_var']

ou melhor: arquivo_h5 = h5py.File(r"C:\Users\User\Downloads\ML\Simulated traning data\N CMAPSS\17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2\17. Turbofan Engine Degradation Simulation Data Set 2\data_set\data_set\N-CMAPSS_DS01-005.h5", 'r')

puxar os nomes das colunas de cada arquivo em em var para transferilos para as tabelas em dev

columnsA  = [columns.decode('utf-8') for columns in arquivo_h5['A_var'][:]]
columnsT  = [columns.decode('utf-8') for columns in arquivo_h5['T_var'][:]]
columnsW  = [columns.decode('utf-8') for columns in arquivo_h5['W_var'][:]]
columnsXs = [columns.decode('utf-8') for columns in arquivo_h5['X_s_var'][:]]
columnsXv = [columns.decode('utf-8') for columns in arquivo_h5['X_v_var'][:]]

df_A  = pd.DataFrame(np.array(arquivo_h5['A_dev']),   columns=columnsA)
df_T  = pd.DataFrame(np.array(arquivo_h5['T_dev']),   columns=columnsT)
df_W  = pd.DataFrame(np.array(arquivo_h5['W_dev']),   columns=columnsW)
df_Xs = pd.DataFrame(np.array(arquivo_h5['X_s_dev']), columns=columnsXs)
df_Xv = pd.DataFrame(np.array(arquivo_h5['X_v_dev']), columns=columnsXv)

rul nao tem var, vamo dar nome manualmente 
df_Y = pd.DataFrame(np.array(arquivo_h5['Y_dev']), columns=['RUL'])

colums da o nome salvo em x_var para a tabela de dados, por fim:

df_completo = pd.concat([df_A,df_T,df_W,df_Xs,df_Xv, df_Y], axis=1)


assim termos uma matriz bidimensional tal que cada linha de A é uma amostra e cada coluna de A é uma feature/atributo

[A][0] = ['id', 'cycle','alt', 'Mach', 'TRA', 'T2', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf', 'T40', 'P30', 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi']

1. A Inicialização (O "Random")
Em vez de chamar de "random input", nós chamamos isso de Inicialização Aleatória de Pesos. Antes do algoritmo começar a aprender, ele cria uma equação vazia: $$\hat{Y} = W \cdot X + b$$ Como o computador ainda é "burro", ele chuta valores completamente aleatórios (random) para o vetor de pesos ($W$) e para o termo independente ($b$). (Nota: $\hat{Y}$ com chapéu significa "Previsão", enquanto $Y$ normal é o Gabarito real).

2. A Previsão (Lendo a linha de $A$)
O algoritmo pega a linha 0 da sua Matriz $A$ (o seu vetor $X$) e multiplica pelos pesos aleatórios ($W$). Como o chute inicial foi aleatório, o $\hat{Y}$ previsto (ex: 50 ciclos) vai ser completamente diferente do $Y$ real (ex: 200 ciclos).

3. A Minimização (O seu "min f(x)")
Na verdade, a função que nós tentamos minimizar (min f(x)) tem um nome especial: Função de Custo (ou Loss Function). A Função de Custo mais famosa é o Erro Quadrático Médio (MSE). Ela calcula a distância entre o que o computador previu ($\hat{Y}$) e o gabarito real ($Y$ - A RUL de verdade).

$$Função\ de\ Custo = (\hat{Y} - Y)^2$$

4. O Aprendizado (Descida do Gradiente)
A partir daí, o algoritmo olha para o Erro e usa o Cálculo (Derivadas/Gradientes) para responder à pergunta: "Como eu preciso alterar os valores de $W$ e $b$ para que esse Erro diminua na próxima vez?"

Ele atualiza o $W$ e o $b$, vai para a linha 1 da Matriz $A$, prevê de novo, calcula o erro, atualiza de novo... até chegar na última linha da Mega Matriz. Quando ele termina de ler toda a Mega Matriz, dizemos que ele completou uma Época (Epoch). Ele faz isso dezenas de vezes até que a Função de Custo chegue no mínimo possível.

Resumo da Ópera: Você descreveu brilhantemente o algoritmo de Gradient Descent. O modelo começa com um hiperplano aleatório ($W, b$), e vai "torcendo" e ajustando a inclinação desse plano a cada linha de sensores ($X$) que ele lê, tentando sempre zerar a distância para a RUL verdadeira ($Y$).


modelo inicial pronto, proxcimo passo é testar C e epsilon infinitos para descobrir o erro base para depois comparar com os outros modelos

podemos usar metodos de calculo numerico para encontrar o ponto otimo de C e epsilon como 
aquele de dividir ao meio, bissecção, não?
axcesso de min locais pode quebrar a logica
usar grid search ou otimização bayesiana
( RESULTADO FINAL DA OTIMIZAÇÃO ---
Menor Erro RMSE Encontrado: 7.09 ciclos
Melhor Configuração Descoberta: {'C': 9750.463241343092, 'epsilon': 1.536250565895301}))
SVR e SVM diferenças
protocolado via CPU isso retardou banstante os calculos do banco de dados completo
exploramos desempenho de diferentes bibliotecas para acelerar o desempenho do calculo

metricas para teste

nSV e o nBSV?

limitar em 5k de linhas, depois em 10K, etc ate o banco completo 
as semples foram retiradas com random para medirmos o impacto das nossas seleçoes de hiperparamentros na diminuiçao do erro medio e erro quadrativo

conforme aumentamos a amostra, os hiperparametros ótimos para o modelo se tornaram menores, o que faz sentido, pois temos mais dados para ajustar o modelo, como consequencia o erro medio e erro quadratico foram caindo progressivamente estabilizando por volta de 7 ciclos.

Começamos a estudar viabilidade de outros algoritimos de ML: random forest pareceu inadequado pela volatilidade e impresivilidade dos parametros de uma aeronave,

 XGBoost (ou LightGBM).

Aqui está o porquê de testá-los:

XGBoost (Extreme Gradient Boosting): Diferente do Random Forest (que cria várias árvores independentes e "vota" no resultado, o que gera imprevisibilidade em séries temporais), o XGBoost é baseado em Gradient Boosting. Ele cria uma árvore pequena e fraca, analisa onde ela errou, e a próxima árvore é focada exclusivamente em corrigir o erro da árvore anterior. Ele vai se "auto-corrigindo" sequencialmente.
Vantagem para Aeronaves: Ele ignora ruídos e "outliers" (picos irreais dos sensores) de forma fantástica, e lida com não-linearidade extrema. É atualmente considerado o "teto máximo" do Machine Learning tradicional.
Super-Trunfo (Performance): Ele treina na base completa muito, mas muito mais rápido que o SVR. Ele possui o parâmetro tree_method='hist', que processa milhões de linhas em segundos na CPU, e possui aceleração nativa na GPU caso necessário.

Etapas para sua Pesquisa:
Regressão Linear / Ridge Regression (O Baseline de Controle): Treine um modelo super burro só para dizer no artigo: "O pior cenário matemático possível nos dá um erro de X ciclos".
SVR (O que você já fez): Provou a vantagem da não-linearidade (kernel rbf), e você já mapeou como as amostras e a Otimização Bayesiana diminuem e estabilizam o erro em 7 ciclos! (Excelente insight no seu Assets.md, por sinal).
XGBoost (O Campeão do ML Clássico): Tentar quebrar esse platô de 7 ciclos usando árvores com aprendizado sequencial.
Deep Learning (LSTMs/CNNs): A cereja do bolo. Justificado porque: "O ML clássico (XGBoost/SVR) estabilizou e não entende a sequência histórica (janela de tempo) do motor. Introduzimos Deep Learning para ler o filme do desgaste e reduzir drasticamente o erro".

conforme nos acostumamos com o formato H5 paramos de convertelos em CSVs com pasndas, optamos por ler os dados diretamente de h5 pyhton, o que otimizou o desempenho do calculo


O resultado do seu teste acabou de comprovar a teoria na prática! O RMSE do Baseline foi de 15.97 ciclos, enquanto o seu SVR conseguiu impressionantes ~7.09 ciclos. O SVR reduziu o erro pela metade.

Aqui estão as diferenças principais para você entender (e até colocar no seu Assets.md):

1. Regressão Normal (Linear Regression)
O que faz: Ela tenta traçar uma linha reta e dura no meio de todos os pontos de dados (equação $Y = aX + b$). O objetivo dela é minimizar o erro de todos os pontos ao mesmo tempo.
Por que falhou (Erro de 15.97): A degradação de uma turbina de avião não é uma linha reta constante. Ela sofre degraus, oscila com o clima (altitude/temperatura) e despenca drasticamente no final. A Regressão Linear é rígida demais para fazer curvas e entender essas nuances.
[Equação Completa da Regressão Linear]
Y (RUL) = 633.7342 (intercepto b)
          + 1.7163 * unit
          - 0.8301 * cycle
          - 8.0097 * Fc
          + 3.9276 * hs
          + 0.0000 * fan_eff_mod
          + 0.0000 * fan_flow_mod
          - 0.0000 * LPC_eff_mod
          - 0.0000 * LPC_flow_mod
          - 0.0000 * HPC_eff_mod
          - 0.0000 * HPC_flow_mod
          + 2323.7396 * HPT_eff_mod
          - 0.0000 * HPT_flow_mod
          + 0.0000 * LPT_eff_mod
          - 0.0000 * LPT_flow_mod
          - 0.0033 * alt
          + 51.1132 * Mach
          - 1.6985 * TRA
          + 0.4621 * T2
          - 0.0876 * T24
          - 1.2203 * T30
          + 2.2801 * T48
          - 0.1137 * T50
          - 12.5217 * P15
          + 14.9304 * P2
          - 12.7124 * P21
          + 5.0757 * P24
          - 31.1103 * Ps30
          + 18.3778 * P40
          + 4.8988 * P50
          + 0.1694 * Nf
          + 0.0352 * Nc
          - 87.5409 * Wf
          - 0.7749 * T40
          + 19.2438 * P30
          - 6.1169 * P45
          - 0.0438 * W21
          - 0.8455 * W22
          + 14.2298 * W25
          + 1.6809 * W31
          + 1.0085 * W32
          - 23.7925 * W48
          + 0.2248 * W50
          + 0.1756 * SmFan
          + 1.8772 * SmLPC
          - 5.4944 * SmHPC
          - 24.1877 * phi

A Maior Penalidade de Quebra (HPT_eff_mod): Note que o peso associado a HPT_eff_mod (Degradação de Eficiência da Turbina de Alta Pressão) é absurdamente alto e positivo (+ 2323.7396). Qualquer variação neste sensor altera o tempo de vida (RUL) da turbina numa escala gigantesca.
Sensores Descartados (Pesos Zerados): Várias variáveis que representam modificações na admissão, como LPC_eff_mod, fan_flow_mod ou HPT_flow_mod, ganharam peso literalmente 0.0000. Isso quer dizer que, pelo menos no "olhar burro" da regressão linear, eles não servem para nada na hora de prever a falha.
Consumo de Combustível (Wf) pesando contra: O Wf ganhou peso negativo (- 87.54). O que faz sentido fisicamente, motores degradados costumam alterar intensamente o consumo ou fluxo, diminuindo severamente as horas restantes até a falha.

2. SVM (Support Vector Machine)
O que faz: Foi criado para Classificação (separar grupos em categorias, tipo "Cachorro" vs "Gato", ou no nosso caso, "Motor Saudável" vs "Motor Falhando").
A Mágica: Em vez de passar uma linha no meio dos pontos, o SVM cria uma "rua" ou fronteira que separa as categorias o mais distante possível. Os pontos que ficam bem na beira da rua são os chamados Vetores de Suporte (Support Vectors).
3. SVR (Support Vector Regression)
O que faz: É a adaptação inteligente do SVM para prever números contínuos (Regressão), como o RUL.
A Diferença: Em vez de tentar passar uma linha perfeita em tudo (como a regressão normal), o SVR cria um "tubo de tolerância" (chamado de $\epsilon$ - epsilon). Ele diz: "Qualquer pequeno erro que caia dentro deste tubo é apenas ruído normal de voo e eu não ligo". Ele só pune pesadamente os erros que fogem muito da realidade (parâmetro $C$).
O Truque de Mestre (Kernel RBF): Ao contrário da Regressão Linear que é obrigada a ser reta, o SVR com Kernel RBF dobra e torce esse "tubo" em múltiplas dimensões matemáticas. Isso permite que ele siga as curvas não-lineares exatas da quebra do motor, ignorando a trepidação leve dos sensores. É por isso que o seu erro caiu de quase 16 para apenas 7 ciclos!


1. MLP (Multi-Layer Perceptron)
O que é: É a rede neural básica (tradicional). Várias camadas de "neurônios" matemáticos interligados.
Prós: Muito fácil de implementar e super rápida de treinar.
Contras: Assim como o XGBoost, ela não tem memória. Ela analisa cada voo isoladamente. Geralmente o resultado de uma MLP simples não consegue superar um XGBoost otimizado.
2. CNN 1D (Redes Neurais Convolucionais 1D)
O que é: As CNNs ficaram famosas por reconhecer imagens e rostos. Mas a versão "1D" (Unidimensional) desliza um filtro não sobre uma imagem, mas sobre a linha do tempo dos sensores. Ela atua como um scanner matemático buscando pequenos "padrões de quebra" (ex: picos súbitos de vibração que duram 5 voos).
Prós: Altíssima velocidade de treinamento (paraleliza muito bem) e excelente para isolar ruídos e identificar anomalias bruscas.
Contras: A CNN enxerga janelas curtas. Ela pode não perceber o desgaste lento e acumulado que ocorre ao longo de centenas de voos inteiros.
3. LSTM (Long Short-Term Memory) / GRU - O Padrão Ouro 🏆
O que é: É o clássico da aviação moderna. Diferente das redes normais, a LSTM possui um "cinto de memória" e "portões" (gates). Ela lê a sequência de voos e decide ativamente o que ela deve lembrar (ex: temperatura subindo aos poucos) e o que deve esquecer (ex: uma turbulência que causou um pico num único voo).
Prós: É a arquitetura soberana para o CMAPSS. Ao injetar uma janela (sliding window) dos últimos 30 voos contínuos da turbina, ela crava a degradação com perfeição.
Contras: São muito lentas para treinar em CPU. É aqui que entra o uso de Placas de Vídeo (GPUs).
4. Transformadores de Séries Temporais (Time-Series Transformers)
O que é: É a arquitetura no estado-da-arte extremo, a mesma matemática revolucionária que faz o ChatGPT funcionar (Mecanismo de Atenção). O Transformer olha para a vida inteira do motor de uma só vez e calcula qual voo no passado tem maior "peso" (Attention) para a quebra que está acontecendo agora.
Prós: Potencial de alcançar o menor erro possível na história desse dataset.
Contras: Extremamente complexo de codificar, exige GPUs poderosas e requer um volume colossal de dados para não sofrer de overfitting.
A Estratégia Recomendada para o Artigo: Geralmente, pesquisas científicas de alto impacto sobre o CMAPSS fazem exatamente este duelo de titãs nas Redes Neurais:

Implementar uma LSTM Clássica (para provar que a introdução do contexto temporal esmaga o Machine Learning clássico).
Opcionalmente, implementar uma Rede Híbrida (CNN + LSTM). A CNN faz a varredura para extrair o ruído, e manda o dado limpo para a LSTM prever o tempo no futuro.


Deep learning
adaptar dataset a tensores 3D

