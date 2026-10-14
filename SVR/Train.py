import h5py
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler


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
    # O SVR tem complexidade espacial e temporal O(n³). Com 5 milhões de linhas, 
    # ele nunca terminaria de treinar em hardware doméstico. Usamos uma amostra.
    # -------------------------------------------------------------------------
    if n_amostra is not None:
        df_treino = df_treino.sample(n=n_amostra, random_state=42)
    
    X_treino = df_treino.drop(columns=['RUL'])
    Y_treino = df_treino['RUL']

    # Calculando Média e Desvio
    media_treino = X_treino.mean()
    desvio_treino = X_treino.std()
    desvio_treino[desvio_treino == 0] = 1.0

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

    # Aplicando Padronização Z-Score usando as métricas do Treino
    X_teste_scaled = (X_teste - media_treino) / desvio_treino

    return X_teste_scaled, Y_teste

# --- CARREGANDO E PADRONIZANDO OS DADOS ---
print("Carregando bases e aplicando Z-Score internamente...")
X_treino_scaled, Y_treino, media_treino, desvio_treino = Matriz_dev(arquivo_h5, n_amostra=20000)
X_teste_scaled, Y_teste = Matriz_test(arquivo_h5, media_treino, desvio_treino)

# C = Penalidade de erro (quanto custa errar fora do tubo)
# epsilon = O raio/grossura do tubo de tolerância a ruídos
# RESULTADO FINAL DA OTIMIZAÇÃO ---
# Menor Erro RMSE Encontrado: 7.09 ciclos
# Melhor Configuração Descoberta: {'C': 9750.463241343092, 'epsilon': 1.536250565895301}

Model = SVR(kernel='rbf', C=9750.463241343092, epsilon=1.536250565895301, verbose = True)

print("Treinando o SVR com dados padronizados...")
Model.fit(X_treino_scaled, Y_treino)

joblib.dump(Model, 'modelo_rul_treinado.pkl')

print("Avaliando o Erro na Base de Teste Cego...")
# Fazer o Predict no Teste
Y_previsto_teste = Model.predict(X_teste_scaled)

# 4. CALCULAR O ERRO REAL (A Avaliação no Teste Cego)
erro_mae = mean_absolute_error(Y_teste, Y_previsto_teste)
print(f"Erro Absoluto Médio (MAE) no Teste: {erro_mae:.2f} ciclos")

erro_mse = mean_squared_error(Y_teste, Y_previsto_teste)
erro_rmse = np.sqrt(erro_mse)
print(f"Erro Quadrático Médio (RMSE) no Teste: {erro_rmse:.2f} ciclos")


