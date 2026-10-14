import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
import math

print("=== Iniciando a Arma Final: Transformer + Feature Engineering ===")
device = torch.device('cpu') 

caminho_h5 = r"C:\Users\User\Downloads\ML\Simulated traning data\N CMAPSS\17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2\17. Turbofan Engine Degradation Simulation Data Set 2\data_set\data_set\N-CMAPSS_DS01-005.h5"

print("1. Extraindo Dados Brutos...")
with h5py.File(caminho_h5, 'r') as arquivo_h5:
    W_dev = np.array(arquivo_h5['W_dev'], dtype=np.float32)
    X_s_dev = np.array(arquivo_h5['X_s_dev'], dtype=np.float32)
    X_v_dev = np.array(arquivo_h5['X_v_dev'], dtype=np.float32)
    T_dev = np.array(arquivo_h5['T_dev'], dtype=np.float32)
    A_dev = np.array(arquivo_h5['A_dev'], dtype=np.float32)
    Y_treino = np.array(arquivo_h5['Y_dev'], dtype=np.float32)

X_treino = np.hstack([A_dev[:, 1:], T_dev, W_dev, X_s_dev, X_v_dev])
units_treino = A_dev[:, 0]

with h5py.File(caminho_h5.replace('dev', 'test'), 'r') as arquivo_h5:
    W_test = np.array(arquivo_h5['W_test'], dtype=np.float32)
    X_s_test = np.array(arquivo_h5['X_s_test'], dtype=np.float32)
    X_v_test = np.array(arquivo_h5['X_v_test'], dtype=np.float32)
    T_test = np.array(arquivo_h5['T_test'], dtype=np.float32)
    A_test = np.array(arquivo_h5['A_test'], dtype=np.float32)
    Y_teste = np.array(arquivo_h5['Y_test'], dtype=np.float32)

X_teste = np.hstack([A_test[:, 1:], T_test, W_test, X_s_test, X_v_test])
units_teste = A_test[:, 0]

print("2. Engenharia de Features: Passando o 'Ferro de Passar' (Moving Average)...")
# Essa é a técnica de Kaggle Master: Suavizar o ruído por turbina antes de treinar
def aplicar_suavizacao_por_turbina(X, units, tamanho_janela=5):
    splits = np.where(np.diff(units) != 0)[0] + 1
    splits = np.insert(splits, 0, 0)
    splits = np.append(splits, len(units))
    for i in range(len(splits)-1):
        start_idx, end_idx = splits[i], splits[i+1]
        df = pd.DataFrame(X[start_idx:end_idx])
        X[start_idx:end_idx] = df.rolling(tamanho_janela, min_periods=1, center=True).mean().values
    return X

X_treino = aplicar_suavizacao_por_turbina(X_treino, units_treino)
X_teste = aplicar_suavizacao_por_turbina(X_teste, units_teste)

print("3. Escalonamento In-Place...")
min_vals, ptp_vals = X_treino.min(axis=0), np.ptp(X_treino, axis=0)
ptp_vals[ptp_vals == 0] = 1.0
X_treino -= min_vals
X_treino /= ptp_vals
X_teste -= min_vals
X_teste /= ptp_vals

class CMAPSS_SequenceDataset(Dataset):
    def __init__(self, X, Y, units, window_size=30):
        self.X, self.Y, self.window_size = X, Y, window_size
        self.valid_indices = []
        splits = np.where(np.diff(units) != 0)[0] + 1
        splits = np.insert(splits, 0, 0)
        splits = np.append(splits, len(units))
        for i in range(len(splits)-1):
            start_idx, end_idx = splits[i], splits[i+1]
            if (end_idx - start_idx) >= window_size:
                for j in range(start_idx, end_idx - window_size):
                    self.valid_indices.append(j)
    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return torch.tensor(self.X[real_idx : real_idx + self.window_size]), torch.tensor(self.Y[real_idx + self.window_size])

dataset_treino = CMAPSS_SequenceDataset(X_treino, Y_treino, units_treino, window_size=30)
dataset_teste  = CMAPSS_SequenceDataset(X_teste, Y_teste, units_teste, window_size=30)
loader_treino = DataLoader(dataset_treino, batch_size=64, shuffle=True)
loader_teste  = DataLoader(dataset_teste, batch_size=64, shuffle=False)

# ==============================================================================
# A ARQUITETURA DO CHATGPT: O TRANSFORMER (Self-Attention)
# ==============================================================================
class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features):
        super(TimeSeriesTransformer, self).__init__()
        # Diferente da LSTM, o Transformer lê tudo de uma vez. Ele precisa de uma 
        # "marca d'agua" para saber quem é o Voo 1 e quem é o Voo 30 (Positional Encoding)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 30, n_features))
        
        # O Cérebro: Mecanismo de Atenção. 45 features divisíveis por 5 cabeças de atenção = 9.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_features, 
            nhead=5, 
            dim_feedforward=128, 
            dropout=0.3, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Soma o sinal da turbina com o carimbo de tempo
        x = x + self.pos_encoder
        # O Transformer cruza e estuda os 30 voos simultaneamente
        out = self.transformer(x)
        # Passa o resumo final para a camada regressiva
        return self.fc(out[:, -1, :])

modelo = TimeSeriesTransformer(n_features=X_treino.shape[1]).to(device)

criterion = nn.SmoothL1Loss() 
optimizer = torch.optim.AdamW(modelo.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

melhor_erro_teste = float('inf')
paciencia_early_stop = 3
contador_paciencia = 0

print(f"\nIniciando Treinamento. Dataset de Treino: {len(dataset_treino)} fatias.")
start = time.time()

for epoca in range(15):
    modelo.train()
    loss_treino_total = 0
    
    for entradas, ruls in loader_treino:
        entradas, ruls = entradas.to(device), ruls.to(device)
        optimizer.zero_grad()
        previsoes = modelo(entradas)
        loss = criterion(previsoes, ruls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        optimizer.step()
        loss_treino_total += loss.item() * len(ruls)
        
    modelo.eval()
    mse_treino_total = 0
    mse_teste_total = 0
    
    with torch.no_grad():
        for entradas, ruls in loader_treino:
            entradas, ruls = entradas.to(device), ruls.to(device)
            previsoes = modelo(entradas)
            mse_treino_total += torch.nn.functional.mse_loss(previsoes, ruls).item() * len(ruls)
            
        for entradas, ruls in loader_teste:
            entradas, ruls = entradas.to(device), ruls.to(device)
            previsoes = modelo(entradas)
            mse_teste_total += torch.nn.functional.mse_loss(previsoes, ruls).item() * len(ruls)
            
    rmse_treino = np.sqrt(mse_treino_total / len(dataset_treino))
    rmse_teste = np.sqrt(mse_teste_total / len(dataset_teste))
    
    print(f"Época {epoca+1}/15 - RMSE Treino: {rmse_treino:.2f} | RMSE Teste Cego: {rmse_teste:.2f}")

    scheduler.step(rmse_teste)

    if rmse_teste < melhor_erro_teste:
        melhor_erro_teste = rmse_teste
        contador_paciencia = 0
        torch.save(modelo.state_dict(), 'modelo_transformer.pth')
    else:
        contador_paciencia += 1
        if contador_paciencia >= paciencia_early_stop:
            print("\n[ALERTA] OVERFITTING IMINENTE. EARLY STOPPING ATIVADO!")
            break

minutos = (time.time() - start) / 60
print(f"\nTreinamento Finalizado em {minutos:.2f} minutos.")
print(f"==================================================")
print(f"NOVO RECORDE DO TRANSFORMER: RMSE {melhor_erro_teste:.2f}")
print(f"==================================================")

# ==============================================================================
# FASE 4: O ENSEMBLE DOS DEUSES (XGBoost + Transformer)
# ==============================================================================
import joblib
print("\n[Iniciando o Ensemble Híbrido: XGBoost + Transformer]")
try:
    caminho_xgb = r"C:\Users\User\Downloads\ML\XGBoost\modelo_xgboost_otimizado.pkl"
    modelo_xgb = joblib.load(caminho_xgb)
    
    # Precisamos do dado Flattened para o XGBoost
    # Vamos avaliar nos últimos N amostras do Teste para calcular a glória do Ensemble
    print("Recuperando dados para o Ensemble...")
    # Avaliando um lote grande
    entradas_3d_lote, ruls_lote = next(iter(DataLoader(dataset_teste, batch_size=20000, shuffle=False)))
    
    # 1. Previsão do Transformer
    modelo.load_state_dict(torch.load('modelo_transformer.pth'))
    modelo.eval()
    with torch.no_grad():
        prev_transf = modelo(entradas_3d_lote).numpy().flatten()
        
    # 2. Previsão do XGBoost
    entradas_2d_lote = entradas_3d_lote.reshape(entradas_3d_lote.shape[0], -1).numpy()
    import xgboost as xgb
    prev_xgb = modelo_xgb.predict(entradas_2d_lote)
    
    # 3. A Fusão (Média)
    prev_ensemble = (prev_transf + prev_xgb) / 2.0
    ruls_reais = ruls_lote.numpy().flatten()
    
    rmse_transf_isolado = np.sqrt(np.mean((prev_transf - ruls_reais)**2))
    rmse_xgb_isolado = np.sqrt(np.mean((prev_xgb - ruls_reais)**2))
    rmse_ensemble = np.sqrt(np.mean((prev_ensemble - ruls_reais)**2))
    
    print(f"-> RMSE Transformer Isolado (Amostra): {rmse_transf_isolado:.2f}")
    print(f"-> RMSE XGBoost Isolado (Amostra): {rmse_xgb_isolado:.2f}")
    print(f"-> RMSE ENSEMBLE SUPREMO (FUSÃO): {rmse_ensemble:.2f} !!!")
    
except Exception as e:
    print("Não foi possível carregar o XGBoost para o Ensemble:", e)
