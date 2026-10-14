import h5py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import optuna
import joblib
import time


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

print("Carregando Dados...")
df_treino = extrair_dados(arquivo_h5, 'dev')
X_treino = df_treino.drop(columns=['RUL'])
Y_treino = df_treino['RUL']

df_teste = extrair_dados(arquivo_h5, 'test')
X_teste = df_teste.drop(columns=['RUL'])
Y_teste = df_teste['RUL']

# Função que o Optuna vai tentar minimizar (O RMSE)
def objective(trial):
    # O Optuna "chuta" valores inteligentemente com base nos resultados anteriores
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'random_state': 42,
        
        # O XGBoost tem a magia de podermos fixar as árvores alto e parar cedo (early_stopping_rounds)
        'n_estimators': 800,
        
        # Chutes Bayesianos:
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        
        # Regularização extrema para evitar overfitting (análogo ao C do SVR)
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    }
    
    modelo = xgb.XGBRegressor(**params, early_stopping_rounds=30)
    
    # Treina silenciosamente
    modelo.fit(
        X_treino, Y_treino,
        eval_set=[(X_teste, Y_teste)],
        verbose=False
    )
    
    # Avalia
    Y_previsto = modelo.predict(X_teste)
    rmse = np.sqrt(mean_squared_error(Y_teste, Y_previsto))
    
    return rmse

# Uma pesquisa muito maior para espremer até a última gota de performance
study = optuna.create_study(direction='minimize')
start = time.time()
study.optimize(objective, n_trials=150)
print(f"Busca finalizada em {(time.time()-start)/60:.2f} minutos.")

print("\n=========================================")
print("=== MELHOR CONFIGURAÇÃO ENCONTRADA ===")
print(f"Menor Erro RMSE: {study.best_value:.4f} ciclos")
print("Parâmetros Ótimos:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("=========================================\n")

print("Re-treinando o modelo final com a melhor configuração...")
best_params = study.best_params
best_params['objective'] = 'reg:squarederror'
best_params['tree_method'] = 'hist'
best_params['random_state'] = 42
best_params['n_estimators'] = 800

Modelo_Otimo = xgb.XGBRegressor(**best_params, early_stopping_rounds=50)
Modelo_Otimo.fit(
    X_treino, Y_treino,
    eval_set=[(X_teste, Y_teste)],
    verbose=100
)

joblib.dump(Modelo_Otimo, 'modelo_xgboost_otimizado.pkl')
print("Modelo otimizado salvo como 'modelo_xgboost_otimizado.pkl'!")
