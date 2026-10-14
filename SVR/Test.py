import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

arquivo_h5 = h5py.File(r"C:\Users\User\Downloads\ML\Simulated traning data\N CMAPSS\17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2\17. Turbofan Engine Degradation Simulation Data Set 2\data_set\data_set\N-CMAPSS_DS01-005.h5", 'r')

# Matriz de treino (Apenas para extrair a Média e Desvio Padrão)
def Matriz_dev_scaler(arquivo_h5):
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
    
    X_treino = df_treino.drop(columns=['RUL'])
    
    media_treino = X_treino.mean()
    desvio_treino = X_treino.std()
    desvio_treino[desvio_treino == 0] = 1.0
    
    return media_treino, desvio_treino

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

    # Aplicando Padronização Z-Score
    X_teste_scaled = (X_teste - media_treino) / desvio_treino

    return X_teste_scaled, Y_teste

# 1. Carregar Métricas de Padronização e Dados de Teste
print("Calculando Scalers do Treino...")
media_treino, desvio_treino = Matriz_dev_scaler(arquivo_h5)

print("Carregando base de Teste...")
X_test_scaled, Y_test = Matriz_test(arquivo_h5, media_treino, desvio_treino)

# 2. Carregar o Modelo Treinado
try:
    Model = joblib.load('modelo_bayesiano_svr_campeao.pkl')
except FileNotFoundError:
    Model = joblib.load('modelo_rul_treinado.pkl')

# 3. Fazer o Predict (Adivinhar as respostas)
Y_previsto = Model.predict(X_test_scaled)

# 4. CALCULAR O ERRO (A Avaliação)
erro_mae = mean_absolute_error(Y_test, Y_previsto)
print(f"Erro Absoluto Médio (MAE): {erro_mae:.2f} ciclos")

erro_mse = mean_squared_error(Y_test, Y_previsto)
erro_rmse = np.sqrt(erro_mse)
print(f"Erro Quadrático Médio (RMSE): {erro_rmse:.2f} ciclos")

# =========================================================
# 5. GRÁFICOS DE AVALIAÇÃO
# =========================================================

# A) Trajetória de Degradação (Série Temporal)
# Plotamos os primeiros 500 ciclos para ilustrar a queda de RUL
plt.figure(figsize=(10, 5))
plt.plot(Y_test.values[:500], label='RUL Real', color='blue', linewidth=2)
plt.plot(Y_previsto[:500], label='RUL Previsto', color='red', linestyle='dashed', linewidth=2)
plt.xlabel('Ciclos (Amostra Temporal)')
plt.ylabel('RUL (Ciclos Restantes)')
plt.title('Trajetória de Degradação (Série Temporal)')
plt.legend()
plt.grid(True)
plt.show()

# B) Gráfico de Dispersão (Real vs. Previsto)
plt.figure(figsize=(8, 8))
plt.scatter(Y_test, Y_previsto, alpha=0.3, color='orange')
max_val = max(Y_test.max(), Y_previsto.max())
plt.plot([0, max_val], [0, max_val], color='blue', linestyle='--', linewidth=2, label='Previsão Perfeita')
plt.xlabel('RUL Real')
plt.ylabel('RUL Previsto')
plt.title('Gráfico de Dispersão (Real vs. Previsto)')
plt.legend()
plt.grid(True)
plt.show()

# C) Distribuição de Erros (Resíduos)
residuos = Y_previsto - Y_test
plt.figure(figsize=(10, 5))
sns.histplot(residuos, bins=50, kde=True, color='purple')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Erro Zero')
plt.xlabel('Erro (RUL Previsto - RUL Real)')
plt.ylabel('Frequência')
plt.title('Distribuição de Erros (Resíduos)')
plt.legend()
plt.grid(True)
plt.show()
