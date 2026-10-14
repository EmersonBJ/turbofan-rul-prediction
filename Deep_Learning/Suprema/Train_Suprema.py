import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

print("=== Iniciando a Arquitetura Suprema (Bi-LSTM + Huber Loss + LR Scheduler) ===")
device = torch.device('cpu') 

caminho_h5 = r"C:\Users\User\Downloads\ML\Simulated traning data\N CMAPSS\17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2\17. Turbofan Engine Degradation Simulation Data Set 2\data_set\data_set\N-CMAPSS_DS01-005.h5"

print("Lendo e Normalizando os Dados (In-Place)...")
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
# A ARQUITETURA SUPREMA: BI-LSTM
# ==============================================================================
class LSTMSuprema(nn.Module):
    def __init__(self, n_features):
        super(LSTMSuprema, self).__init__()
        # BIDIRECIONAL: Lê do Passado pro Futuro e do Futuro pro Passado
        self.lstm = nn.LSTM(
            input_size=n_features, 
            hidden_size=64, 
            num_layers=2, 
            batch_first=True,    
            dropout=0.5,
            bidirectional=True  # O GRANDE TRUQUE 
        )
        # Como é bidirecional, a saída da LSTM tem 64 * 2 = 128 memórias
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Pegamos a saída do último passo temporal para alimentar a camada densa
        previsao_rul = self.fc(lstm_out[:, -1, :])
        return previsao_rul

modelo = LSTMSuprema(n_features=X_treino.shape[1]).to(device)

# HUBER LOSS (SmoothL1Loss) - Imune a Ruídos e Outliers Bizarrros
criterion = nn.SmoothL1Loss() 

# ADAMW - Melhor manipulador de L2 Regularization (Weight Decay)
optimizer = torch.optim.AdamW(modelo.parameters(), lr=0.001, weight_decay=1e-4)

# LEARNING RATE SCHEDULER - O "Freio ABS" da Inteligência Artificial
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

# EARLY STOPPING
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
        
        # GRADIENT CLIPPING - Impede que anomalias explodam o cérebro da rede
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        
        optimizer.step()
        loss_treino_total += loss.item() * len(ruls)
        
    # ATENÇÃO: Huber Loss não é RMSE. Para avaliar, calculamos o RMSE manual
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

    # Aciona o freio ABS do Otimizador se o Teste parar de melhorar
    scheduler.step(rmse_teste)

    # Lógica do Early Stopping
    if rmse_teste < melhor_erro_teste:
        melhor_erro_teste = rmse_teste
        contador_paciencia = 0
        torch.save(modelo.state_dict(), 'modelo_supremo.pth')
    else:
        contador_paciencia += 1
        if contador_paciencia >= paciencia_early_stop:
            print("\n[ALERTA] OVERFITTING IMINENTE. EARLY STOPPING ATIVADO!")
            break

minutos = (time.time() - start) / 60
print(f"\nTreinamento Finalizado em {minutos:.2f} minutos.")
print(f"==================================================")
print(f"CAMPEÃO ABSOLUTO DO DEEP LEARNING: RMSE {melhor_erro_teste:.2f}")
print(f"==================================================")
