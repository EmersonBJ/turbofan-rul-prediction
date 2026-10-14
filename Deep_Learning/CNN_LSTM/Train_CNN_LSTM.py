import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
import gc

print("=== Iniciando Pipeline Híbrido CNN+LSTM (PyTorch) ===")

device = torch.device('cpu')
print(f"Hardware selecionado: {device} (Modo Segurança de Memória)")

caminho_h5 = r"C:\Users\User\Downloads\ML\Simulated traning data\N CMAPSS\17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2\17. Turbofan Engine Degradation Simulation Data Set 2\data_set\data_set\N-CMAPSS_DS01-005.h5"
arquivo_h5 = h5py.File(caminho_h5, 'r')

def extrair_dados_numpy(arquivo, prefixo='dev'):
    A = np.array(arquivo[f'A_{prefixo}'], dtype=np.float32)
    T = np.array(arquivo[f'T_{prefixo}'], dtype=np.float32)
    W = np.array(arquivo[f'W_{prefixo}'], dtype=np.float32)
    Xs = np.array(arquivo[f'X_s_{prefixo}'], dtype=np.float32)
    Xv = np.array(arquivo[f'X_v_{prefixo}'], dtype=np.float32)
    Y = np.array(arquivo[f'Y_{prefixo}'], dtype=np.float32)
    
    unit_array = A[:, 0]
    features = np.hstack([A[:, 1:], T, W, Xs, Xv])
    
    return features, Y, unit_array

print("Extraindo matrizes do H5...")
X_treino, Y_treino, units_treino = extrair_dados_numpy(arquivo_h5, 'dev')
X_teste, Y_teste, units_teste    = extrair_dados_numpy(arquivo_h5, 'test')
arquivo_h5.close()
gc.collect()

print("Normalização In-Place...")
min_vals = X_treino.min(axis=0)
max_vals = X_treino.max(axis=0)
range_vals = max_vals - min_vals
range_vals[range_vals == 0] = 1.0

X_treino -= min_vals
X_treino /= range_vals
X_teste -= min_vals
X_teste /= range_vals

class CMAPSS_SequenceDataset(Dataset):
    def __init__(self, X, Y, units, window_size=30):
        self.X = X
        self.Y = Y
        self.window_size = window_size
        mudancas = np.where(units[:-1] != units[1:])[0] + 1
        splits = [0] + mudancas.tolist() + [len(units)]
        
        self.valid_indices = []
        for i in range(len(splits) - 1):
            start_idx = splits[i]
            end_idx = splits[i+1]
            if (end_idx - start_idx) >= window_size:
                for j in range(start_idx, end_idx - window_size):
                    self.valid_indices.append(j)
                    
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        seq_x = self.X[real_idx : real_idx + self.window_size]
        seq_y = self.Y[real_idx + self.window_size]
        return torch.tensor(seq_x), torch.tensor(seq_y)

WINDOW_SIZE = 30
dataset_treino = CMAPSS_SequenceDataset(X_treino, Y_treino, units_treino, window_size=WINDOW_SIZE)
dataset_teste  = CMAPSS_SequenceDataset(X_teste, Y_teste, units_teste, window_size=WINDOW_SIZE)

loader_treino = DataLoader(dataset_treino, batch_size=64, shuffle=True)
loader_teste  = DataLoader(dataset_teste, batch_size=64, shuffle=False)

# === A MÁGICA DA REDE HÍBRIDA (CNN + LSTM) ===
class CNN_LSTMPredictor(nn.Module):
    def __init__(self, n_features):
        super(CNN_LSTMPredictor, self).__init__()
        
        # 1. CNN (O Extrator de Ruído)
        # Varre a janela de 30 voos e acha padrões curtos, reduzindo o tamanho pela metade
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 2. LSTM (A Fita de Memória)
        # A CNN entregou 64 canais (features filtradas). A LSTM assume a partir daqui.
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 3. Decisor Final
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Passo 1: Preparar para CNN [Batch, Sensores, Tempo]
        x = x.transpose(1, 2)
        
        # Passo 2: CNN mastiga o dado
        x = self.cnn(x)
        
        # Passo 3: Preparar para LSTM [Batch, Tempo, Features Limpas]
        x = x.transpose(1, 2)
        
        # Passo 4: LSTM analisa a evolução cronológica
        lstm_out, _ = self.lstm(x)
        
        # Passo 5: Pegamos a última conclusão da LSTM e jogamos na saída
        ultima_memoria = lstm_out[:, -1, :]
        previsao_rul = self.fc(ultima_memoria)
        return previsao_rul

n_sensores = X_treino.shape[1]
modelo = CNN_LSTMPredictor(n_features=n_sensores).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)

EPOCHS = 10
print("\nIniciando Treinamento do Duelo de Titãs (CNN + LSTM)...")
start = time.time()

historico_treino = []
historico_teste = []

for epoch in range(EPOCHS):
    # --- FASE DE TREINO ---
    modelo.train()
    loss_treino_acumulada = 0
    for batch_x, batch_y in loader_treino:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        previsoes = modelo(batch_x)
        loss = criterion(previsoes, batch_y.view(-1, 1))
        loss.backward()
        optimizer.step()
        loss_treino_acumulada += loss.item()
        
    rmse_treino = np.sqrt(loss_treino_acumulada / len(loader_treino))
    historico_treino.append(rmse_treino)
    
    # --- FASE DE VALIDAÇÃO ---
    modelo.eval()
    loss_teste_acumulada = 0
    with torch.no_grad():
        for batch_x, batch_y in loader_teste:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            previsoes = modelo(batch_x)
            loss_teste = criterion(previsoes, batch_y.view(-1, 1))
            loss_teste_acumulada += loss_teste.item()
            
    rmse_teste = np.sqrt(loss_teste_acumulada / len(loader_teste))
    historico_teste.append(rmse_teste)
    
    print(f"Época {epoch+1}/{EPOCHS} - RMSE Treino: {rmse_treino:.2f} | RMSE Teste: {rmse_teste:.2f}")

print(f"Treinamento Híbrido concluído em {(time.time()-start)/60:.2f} minutos!")
torch.save(modelo.state_dict(), 'modelo_hibrido.pth')

# === GRÁFICO DA CURVA DE APRENDIZADO ===
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS+1), historico_treino, label='Erro de Treino', color='blue', marker='o')
plt.plot(range(1, EPOCHS+1), historico_teste, label='Erro de Teste (Validação)', color='red', marker='s')
plt.xlabel('Épocas')
plt.ylabel('RMSE (Ciclos de Erro)')
plt.title('Curva de Aprendizado Híbrida CNN+LSTM (Overfitting)')
plt.legend()
plt.grid(True)
plt.savefig('grafico_hibrido_learning_curve.png', bbox_inches='tight')
print("Gráfico da Curva de Aprendizado gerado com sucesso!")
print("\nTestando o modelo Híbrido Cego...")
modelo.eval()
Y_previsto_lista, Y_real_lista = [], []

with torch.no_grad():
    for batch_x, batch_y in loader_teste:
        batch_x = batch_x.to(device)
        previsoes = modelo(batch_x)
        Y_previsto_lista.extend(previsoes.cpu().numpy())
        Y_real_lista.extend(batch_y.numpy())

Y_previsto = np.array(Y_previsto_lista).flatten()
Y_real = np.array(Y_real_lista).flatten()

erro_mae = mean_absolute_error(Y_real, Y_previsto)
erro_rmse = np.sqrt(mean_squared_error(Y_real, Y_previsto))

print("\n=== RESULTADOS HÍBRIDOS (CNN + LSTM) ===")
print(f"Erro Absoluto Médio (MAE): {erro_mae:.2f} ciclos")
print(f"Erro Quadrático Médio (RMSE): {erro_rmse:.2f} ciclos")
print("========================================")

plt.figure(figsize=(10, 5))
plt.plot(Y_real[:500], label='RUL Real', color='blue', linewidth=2)
plt.plot(Y_previsto[:500], label='RUL Previsto (CNN+LSTM)', color='red', linestyle='dashed', linewidth=2)
plt.xlabel('Janelas Temporais')
plt.ylabel('RUL')
plt.title('Trajetória de Degradação (O Duelo Final)')
plt.legend()
plt.grid(True)
plt.savefig('grafico_hibrido_trajetoria.png', bbox_inches='tight')
print("Gráfico salvo!")
