import h5py
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import optuna

arquivo_h5 = h5py.File(r"C:\Users\User\Downloads\ML\Simulated traning data\N CMAPSS\17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2\17. Turbofan Engine Degradation Simulation Data Set 2\data_set\data_set\N-CMAPSS_DS01-005.h5", 'r')

# Matriz de treino (Padronizada)
def Matriz_dev(arquivo_h5, n_amostra=20000):
    columns_A  = [columns.decode('utf-8') for columns in arquivo_h5['A_var'][:]]
    columns_T  = [columns.decode('utf-8') for columns in arquivo_h5['T_var'][:]]
    columns_W  = [columns.decode('utf-8') for columns in arquivo_h5['W_var'][:]]
    columns_Xs = [columns.decode('utf-8') for columns in arquivo_h5['X_s_var'][:]]
    columns_Xv = [columns.decode('utf-8') for columns in arquivo_h5['X_v_var'][:]]

    df_A  = pd.DataFrame(np.array(arquivo_h5['A_dev']),   columns=columns_A)
    df_T  = pd.DataFrame(np.array(arquivo_h5['T_dev']),   columns=columns_T)
    df_W  = pd.DataFrame(np.array(arquivo_h5['W_dev']),   columns=columns_W)
    df_Xs = pd.DataFrame(np.array(arquivo_h5['X_s_dev']), columns=columns_Xs)
    df_Xv = pd.DataFrame(np.array(arquivo_h5['X_v_dev']), columns=columns_Xv)
    df_Y  = pd.DataFrame(np.array(arquivo_h5['Y_dev']),   columns=['RUL'])

    df_treino = pd.concat([df_A, df_T, df_W, df_Xs, df_Xv, df_Y], axis=1)

    # -------------------------------------------------------------------------
    # AMOSTRAGEM (SAMPLE) - EXPLICAÇÃO CRÍTICA
    # Por que não rodar o "pacote completo" (5 milhões de linhas)?
    # O Algoritmo SVR calcula a distância de CADA ponto para TODOS os outros pontos (O(n²)).
    # Rodar 5 milhões de linhas travaria qualquer computador comum por semanas/meses.
    # Usamos uma amostra (ex: 20000) para treinar rápido. Se quiser rodar com TUDO,
    # basta colocar n_amostra=None na chamada da função (ou comentar a linha abaixo).
    # -------------------------------------------------------------------------
    if n_amostra is not None:
        df_treino = df_treino.sample(n=n_amostra, random_state=42)
    
    X_treino = df_treino.drop(columns=['RUL'])
    Y_treino = df_treino['RUL']

    # Calculando a métrica do Z-Score AQUI DENTRO (Média e Desvio do Treino)
    media_treino = X_treino.mean()
    desvio_treino = X_treino.std()
    desvio_treino[desvio_treino == 0] = 1.0

    # Aplicando Padronização Z-Score na mão (Valor - Média / Desvio)
    X_treino_scaled = (X_treino - media_treino) / desvio_treino

    return X_treino_scaled, Y_treino, media_treino, desvio_treino

# Matriz de teste (Padronizada)
def Matriz_test(arquivo_h5, media_treino, desvio_treino):
    columns_A  = [columns.decode('utf-8') for columns in arquivo_h5['A_var'][:]]
    columns_T  = [columns.decode('utf-8') for columns in arquivo_h5['T_var'][:]]
    columns_W  = [columns.decode('utf-8') for columns in arquivo_h5['W_var'][:]]
    columns_Xs = [columns.decode('utf-8') for columns in arquivo_h5['X_s_var'][:]]
    columns_Xv = [columns.decode('utf-8') for columns in arquivo_h5['X_v_var'][:]]

    df_A  = pd.DataFrame(np.array(arquivo_h5['A_test']),   columns=columns_A)
    df_T  = pd.DataFrame(np.array(arquivo_h5['T_test']),   columns=columns_T)
    df_W  = pd.DataFrame(np.array(arquivo_h5['W_test']),   columns=columns_W)
    df_Xs = pd.DataFrame(np.array(arquivo_h5['X_s_test']), columns=columns_Xs)
    df_Xv = pd.DataFrame(np.array(arquivo_h5['X_v_test']), columns=columns_Xv)
    df_Y  = pd.DataFrame(np.array(arquivo_h5['Y_test']),   columns=['RUL'])

    df_test = pd.concat([df_A, df_T, df_W, df_Xs, df_Xv, df_Y], axis=1)
    
    X_teste = df_test.drop(columns=['RUL'])
    Y_teste = df_test['RUL']

    # Aplicando Padronização Z-Score usando as métricas APRENDIDAS NO TREINO (Evita Data Leakage)
    X_teste_scaled = (X_teste - media_treino) / desvio_treino

    return X_teste_scaled, Y_teste

# --- CARREGANDO E PADRONIZANDO OS DADOS ---
# Tudo sai padronizado e pronto para uso direto das funções!
print("Carregando bases e aplicando Z-Score internamente...")
X_treino_scaled, Y_treino, media_treino, desvio_treino = Matriz_dev(arquivo_h5, n_amostra=20000)
X_teste_scaled, Y_teste = Matriz_test(arquivo_h5, media_treino, desvio_treino)

# --- OTIMIZAÇÃO BAYESIANA COM OPTUNA ---

print("\n[OPTUNA] Iniciando Otimização Bayesiana para SVR...")

# O Optuna precisa de uma função que ele vai tentar minimizar
def objective(trial):
    # 1. O robô chuta valores matematicamente estratégicos
    C = trial.suggest_float("C", 0.1, 10000.0, log=True)
    eps = trial.suggest_float("epsilon", 0.05, 2.0, log=True)
    
    # 2. Cria e treina a máquina com os DADOS PADRONIZADOS
    Model = SVR(kernel='rbf', C=C, epsilon=eps)
    Model.fit(X_treino_scaled, Y_treino)
    
    # 3. Calcula o Erro NA BASE DE TESTE CEGO para não overfittar os hiperparâmetros!
    Y_previsto = Model.predict(X_teste_scaled)
    rmse = np.sqrt(mean_squared_error(Y_teste, Y_previsto))
    
    # 4. Devolve o erro real para o Optuna julgar
    return rmse

# Criar um "Estudo" cujo objetivo é MINIMIZAR o erro
study = optuna.create_study(direction="minimize")

# Rodar tentativas (trials). Para testes rápidos, pode baixar para 10 ou 30.
study.optimize(objective, n_trials=30)

print("\n--- RESULTADO FINAL DA OTIMIZAÇÃO ---")
print(f"Menor Erro RMSE Encontrado: {study.best_value:.2f} ciclos")
print(f"Melhor Configuração Descoberta: {study.best_params}")

# --- TREINAR O MODELO VENCEDOR ---
print("\nTreinando o modelo definitivo com os melhores hiperparâmetros...")
melhor_C = study.best_params["C"]
melhor_eps = study.best_params["epsilon"]

Model_Vencedor = SVR(kernel='rbf', C=melhor_C, epsilon=melhor_eps)
Model_Vencedor.fit(X_treino_scaled, Y_treino)

# Salvar o modelo campeão
joblib.dump(Model_Vencedor, 'modelo_rul_campeao.pkl')
print("Modelo Campeão salvo como 'modelo_rul_campeao.pkl'!")


