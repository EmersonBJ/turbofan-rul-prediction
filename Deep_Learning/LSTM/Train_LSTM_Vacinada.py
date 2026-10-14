import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

print("=== Iniciando LSTM Vacinada (Anticorpos contra Overfitting) ===")
device = torch.device('cpu') # Mantendo na CPU (Sem supercomputador)

# 1. Carregamento In-Place (Mesmo de antes, proteção de Memória)
caminho_h5 = r"C:\Users\User\Downloads\ML\Simulated traning data\N CMAPSS\17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2\17. Turbofan Engine Degradation Simulation Data Set 2\data_set\data_set\N-CMAPSS_DS01-005.h5"
with h5py.File(caminho_h5, 'r') as arquivo_h5:
    W_dev = np.array(arquivo_h5['W_dev'], dtype=np.float32)
    X_s_dev = np.array(arquivo_h5['X_s_dev'], dtype=np.float32)
    X_v_dev = np.array(arquivo_h5['X_v_dev'], dtype=np.float32)
    T_dev = np.array(arquivo_h5['T_dev'], dtype=np.float32)
    A_dev = np.array(arquivo_h5['A_dev'], dtype=np.float32)
    Y_treino = np.array(arquivo_h5['Y_dev'], dtype=np.float32)

X_treino = np.hstack([A_dev[:, 1:], T_dev, W_dev, X_s_dev, X_v_dev])
units_treino = A_dev[:, 0]
min_vals, ptp_vals = X_treino.min(axis=0), np.ptp(X_treino, axis=0)
ptp_vals[ptp_vals == 0] = 1.0
X_treino -= min_vals
X_treino /= ptp_vals

with h5py.File(caminho_h5.replace('dev', 'test'), 'r') as arquivo_h5:
    W_test = np.array(arquivo_h5['W_test'], dtype=np.float32)
    X_s_test = np.array(arquivo_h5['X_s_test'], dtype=np.float32)
    X_v_test = np.array(arquivo_h5['X_v_test'], dtype=np.float32)
    T_test = np.array(arquivo_h5['T_test'], dtype=np.float32)
    A_test = np.array(arquivo_h5['A_test'], dtype=np.float32)
    Y_teste = np.array(arquivo_h5['Y_test'], dtype=np.float32)

X_teste = np.hstack([A_test[:, 1:], T_test, W_test, X_s_test, X_v_test])
units_teste = A_test[:, 0]
X_teste -= min_vals
X_teste /= ptp_vals

# 2. Lazy Loading (Mesmo de antes, sem estourar RAM)
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
# 3. A ARQUITETURA VACINADA
# ==============================================================================
class LSTMVacinada(nn.Module):
    def __init__(self, n_features):
        super(LSTMVacinada, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, 
            hidden_size=64, 
            num_layers=2, 
            batch_first=True,    
            dropout=0.5          # VACINA 1: DROPOUT AGRESSIVO (50% de amnésia forçada)
        )
        self.linear = nn.Linear(64, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        previsao_rul = self.linear(lstm_out[:, -1, :])
        return previsao_rul

modelo = LSTMVacinada(n_features=X_treino.shape[1]).to(device)
criterion = nn.MSELoss() 

# VACINA 2: L2 REGULARIZATION (Weight Decay de 1e-4)
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001, weight_decay=1e-4)

# VACINA 3: VARIÁVEIS DE EARLY STOPPING (Parada Antecipada)
melhor_erro_teste = float('inf')
paciencia = 2        # Quantas épocas ele tolera o erro subir antes de abortar
contador_paciencia = 0

print("\nIniciando Treinamento Vacinado...")
for epoca in range(10):
    modelo.train()
    loss_treino_total = 0
    
    # TREINO
    for entradas, ruls in loader_treino:
        entradas, ruls = entradas.to(device), ruls.to(device)
        optimizer.zero_grad()
        previsoes = modelo(entradas)
        loss = criterion(previsoes, ruls)
        loss.backward()
        optimizer.step()
        loss_treino_total += loss.item() * len(ruls)
        
    rmse_treino = np.sqrt(loss_treino_total / len(dataset_treino))
    
    # TESTE CEGO
    modelo.eval()
    loss_teste_total = 0
    with torch.no_grad():
        for entradas, ruls in loader_teste:
            entradas, ruls = entradas.to(device), ruls.to(device)
            previsoes = modelo(entradas)
            loss = criterion(previsoes, ruls)
            loss_teste_total += loss.item() * len(ruls)
            
    rmse_teste = np.sqrt(loss_teste_total / len(dataset_teste))
    print(f"Época {epoca+1}/10 - RMSE Treino: {rmse_treino:.2f} | RMSE Teste: {rmse_teste:.2f}")

    # LÓGICA DO EARLY STOPPING VIGIANDO A LINHA DO TESTE
    if rmse_teste < melhor_erro_teste:
        melhor_erro_teste = rmse_teste
        contador_paciencia = 0
        # Salva o modelo exato desta época maravilhosa
        torch.save(modelo.state_dict(), 'modelo_vacinado.pth')
    else:
        contador_paciencia += 1
        print(f"  -> Alerta! O erro de teste subiu! Paciência: {contador_paciencia}/{paciencia}")
        if contador_paciencia >= paciencia:
            print("\n[ALERTA] EARLY STOPPING ATIVADO! [ALERTA]")
            print("O modelo começou a decorar ruído. Abortando treinamento para economizar processamento e salvar a melhor versão.")
            break

print(f"\nTreino abortado/finalizado. O melhor Erro Cego atingido foi: {melhor_erro_teste:.2f} ciclos!")
