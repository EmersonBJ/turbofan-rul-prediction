import h5py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho para o seu arquivo .h5
caminho_h5 = r"C:\Users\User\Downloads\ML\Simulated traning data\N CMAPSS\17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2\17. Turbofan Engine Degradation Simulation Data Set 2\data_set\data_set\N-CMAPSS_DS01-005.h5"
arquivo_h5 = h5py.File(caminho_h5, 'r')

def extrair_dados(arquivo_h5, prefixo='dev'):
    # prefixo 'dev' para treino e 'test' para teste
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

# 1. Carregando os Dados
df_treino = extrair_dados(arquivo_h5, 'dev')
X_treino = df_treino.drop(columns=['RUL'])
Y_treino = df_treino['RUL']

df_teste = extrair_dados(arquivo_h5, 'test')
X_teste = df_teste.drop(columns=['RUL'])
Y_teste = df_teste['RUL']

# 2. Treinando o Modelo (Y = aX + b)
# A formulinha padrão de regressão que traça uma linha reta pelos dados
# Diferente do SVR, a Regressão Linear resolve a matemática instantaneamente, 
# então podemos treinar na base INTEIRA sem problema de lentidão!
Modelo_Linear = LinearRegression()
Modelo_Linear.fit(X_treino, Y_treino)

# Salvando o modelo caso queira reaproveitá-lo depois
joblib.dump(Modelo_Linear, 'modelo_baseline_linear.pkl')

# 3. Avaliação no Teste
print("Calculando previsões no Teste...")
Y_previsto = Modelo_Linear.predict(X_teste)

erro_mae = mean_absolute_error(Y_teste, Y_previsto)
erro_mse = mean_squared_error(Y_teste, Y_previsto)
erro_rmse = np.sqrt(erro_mse)

# Salvar a tabela de erros
tabela_erros = pd.DataFrame({'Metrica': ['MAE', 'RMSE'], 'Valor (Ciclos)': [erro_mae, erro_rmse]})
tabela_erros.to_csv('tabela_erros_linear.csv', index=False)
print("-> Tabela de erros salva em 'tabela_erros_linear.csv'")

print("\n=== RESULTADOS DA REGRESSÃO LINEAR (BASELINE) ===")
print(f"Erro Absoluto Médio (MAE): {erro_mae:.2f} ciclos")
print(f"Erro Quadrático Médio (RMSE): {erro_rmse:.2f} ciclos")
print("=================================================")

# Como bônus pedagógico, os coeficientes mostram a "formulinha" matemática da aula:
print("\n[Equação Completa da Regressão Linear]")
equacao = f"Y (RUL) = {Modelo_Linear.intercept_:.4f} (intercepto b)"
for nome_coluna, coeficiente in zip(X_treino.columns, Modelo_Linear.coef_):
    sinal = "+" if coeficiente >= 0 else "-"
    equacao += f"\n          {sinal} {abs(coeficiente):.4f} * {nome_coluna}"

print(equacao)

# =========================================================
# 5. GRÁFICOS DE AVALIAÇÃO (COMPARAÇÃO VISUAL DO ERRO)
# =========================================================

print("\nGerando gráficos...")

# A) Trajetória de Degradação (Série Temporal)
plt.figure(figsize=(10, 5))
plt.plot(Y_teste.values[:500], label='RUL Real', color='blue', linewidth=2)
plt.plot(Y_previsto[:500], label='RUL Previsto (Linear)', color='red', linestyle='dashed', linewidth=2)
plt.xlabel('Ciclos (Amostra Temporal)')
plt.ylabel('RUL (Ciclos Restantes)')
plt.title('Trajetória de Degradação: A linha reta tentando imitar a curva')
plt.legend()
plt.grid(True)
plt.savefig('grafico_1_trajetoria_linear.png', bbox_inches='tight')
plt.show()

# B) Gráfico de Dispersão (Real vs. Previsto)
plt.figure(figsize=(8, 8))
plt.scatter(Y_teste, Y_previsto, alpha=0.3, color='orange')
max_val = max(Y_teste.max(), Y_previsto.max())
plt.plot([0, max_val], [0, max_val], color='blue', linestyle='--', linewidth=2, label='Previsão Perfeita')
plt.xlabel('RUL Real')
plt.ylabel('RUL Previsto (Linear)')
plt.title('Dispersão (Real vs. Previsto) - Regressão Linear')
plt.legend()
plt.grid(True)
plt.savefig('grafico_2_dispersao_linear.png', bbox_inches='tight')
plt.show()

# C) Distribuição de Erros (Resíduos)
residuos = Y_previsto - Y_teste
plt.figure(figsize=(10, 5))
sns.histplot(residuos, bins=50, kde=True, color='purple')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Erro Zero')
plt.xlabel('Erro (RUL Previsto - RUL Real)')
plt.ylabel('Frequência')
plt.title('Distribuição de Erros (Resíduos) - Regressão Linear')
plt.legend()
plt.grid(True)
plt.savefig('grafico_3_residuos_linear.png', bbox_inches='tight')
plt.show()
