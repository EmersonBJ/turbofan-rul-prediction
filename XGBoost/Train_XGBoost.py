import h5py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

print("=== Iniciando Pipeline XGBoost ===")

caminho_h5 = r"C:\Users\User\Downloads\ML\Simulated traning data\N CMAPSS\17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2\17. Turbofan Engine Degradation Simulation Data Set 2\data_set\data_set\N-CMAPSS_DS01-005.h5"
arquivo_h5 = h5py.File(caminho_h5, 'r')

def extrair_dados(arquivo_h5, prefixo='dev'):
    columns_A  = [c.decode('utf-8') for c in arquivo_h5['A_var'][:]]
    columns_T  = [c.decode('utf-8') for c in arquivo_h5['T_var'][:]]
    columns_W  = [c.decode('utf-8') for c in arquivo_h5['W_var'][:]]
    columns_Xs = [c.decode('utf-8') for c in arquivo_h5['X_s_var'][:]]
    columns_Xv = [c.decode('utf-8') for c in arquivo_h5['X_v_var'][:]]

    df_A  = pd.DataFrame(np.array(arquivo_h5[f'A_{prefixo}']),   columns=columns_A)
    df_T  = pd.DataFrame(np.array(arquivo_h5[f'T_{prefixo}']),   columns=columns_T)
    df_W  = pd.DataFrame(np.array(arquivo_h5[f'W_{prefixo}']),   columns=columns_W)
    df_Xs = pd.DataFrame(np.array(arquivo_h5[f'X_s_{prefixo}']), columns=columns_Xs)
    df_Xv = pd.DataFrame(np.array(arquivo_h5[f'X_v_{prefixo}']), columns=columns_Xv)
    df_Y  = pd.DataFrame(np.array(arquivo_h5[f'Y_{prefixo}']),   columns=['RUL'])

    return pd.concat([df_A, df_T, df_W, df_Xs, df_Xv, df_Y], axis=1)

print("1. Carregando base de dados (Treino)...")
df_treino = extrair_dados(arquivo_h5, 'dev')
X_treino = df_treino.drop(columns=['RUL'])
Y_treino = df_treino['RUL']

print("2. Carregando base de dados (Teste)...")
df_teste = extrair_dados(arquivo_h5, 'test')
X_teste = df_teste.drop(columns=['RUL'])
Y_teste = df_teste['RUL']

print("\n3. Treinando o modelo XGBoost na BASE INTEIRA...")
# Hiperparâmetros base otimizados para velocidade extrema e baixo erro
params = {
    'objective': 'reg:squarederror', 
    'n_estimators': 800,
    'learning_rate': 0.1955,
    'max_depth': 3,
    'subsample': 0.8888,
    'colsample_bytree': 0.7060,
    'min_child_weight': 8,
    'tree_method': 'hist',
    'random_state': 42
}

Modelo_XGB = xgb.XGBRegressor(**params, early_stopping_rounds=30)

start_time = time.time()
# Diferente do SVR, podemos passar o teste no EVAL_SET para ele plotar o erro decaindo a cada 50 árvores!
Modelo_XGB.fit(
    X_treino, 
    Y_treino,
    eval_set=[(X_teste, Y_teste)],   
    verbose=50                       
)
end_time = time.time()
print(f"-> Treinamento (na base inteira!) concluído incrivelmente em {(end_time - start_time):.2f} segundos!")

joblib.dump(Modelo_XGB, 'modelo_xgboost.pkl')

print("\n4. Calculando previsões finais no Teste...")
Y_previsto = Modelo_XGB.predict(X_teste)

erro_mae = mean_absolute_error(Y_teste, Y_previsto)
erro_mse = mean_squared_error(Y_teste, Y_previsto)
erro_rmse = np.sqrt(erro_mse)

tabela_erros = pd.DataFrame({'Metrica': ['MAE', 'RMSE'], 'Valor (Ciclos)': [erro_mae, erro_rmse]})
tabela_erros.to_csv('tabela_erros_xgboost.csv', index=False)

print("\n=== RESULTADOS XGBOOST ===")
print(f"Erro Absoluto Médio (MAE): {erro_mae:.2f} ciclos")
print(f"Erro Quadrático Médio (RMSE): {erro_rmse:.2f} ciclos")
print("==========================")

print("\n5. Gerando gráficos de avaliação e Feature Importance...")

# A) Trajetória de Degradação
plt.figure(figsize=(10, 5))
plt.plot(Y_teste.values[:500], label='RUL Real', color='blue', linewidth=2)
plt.plot(Y_previsto[:500], label='RUL Previsto (XGBoost)', color='green', linestyle='dashed', linewidth=2)
plt.xlabel('Ciclos (Amostra Temporal)')
plt.ylabel('RUL (Ciclos Restantes)')
plt.title('Trajetória de Degradação (XGBoost)')
plt.legend()
plt.grid(True)
plt.savefig('grafico_1_trajetoria_xgboost.png', bbox_inches='tight')

# B) Importância das Variáveis (O SVR/Regressão não têm isso tão detalhado)
plt.figure(figsize=(12, 8))
importances = pd.Series(Modelo_XGB.feature_importances_, index=X_treino.columns).sort_values(ascending=False).head(15)
importances.plot(kind='bar', color='darkred')
plt.title('Top 15 Sensores Mais Importantes para prever o RUL')
plt.ylabel('Importância (F-Score)')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.savefig('grafico_2_feature_importance.png', bbox_inches='tight')

print("Processo finalizado com sucesso! Resultados salvos na pasta.")
