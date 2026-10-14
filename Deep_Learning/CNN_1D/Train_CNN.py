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

print("=== Iniciando Pipeline CNN 1D (PyTorch) ===")

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

# === A MÁGICA DA CNN 1D ===
class CNN1DPredictor(nn.Module):
    def __init__(self, n_features):
        super(CNN1DPredictor, self).__init__()
        # A CNN procura padrões (filtros/kernels) deslizando no tempo
        self.conv_layer = nn.Sequential(
            # nn.Conv1d precisa de entrada [Batch, Canais (Sensores), Tempo]
            nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Reduz a janela pela metade
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Achatamos os filtros e prevemos a RUL
        # O tamanho após dois MaxPools de uma janela de 30 é: 30 -> 15 -> 7
        self.flatten = nn.Flatten()
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # A LSTM lia [Batch, Tempo, Sensores]. A CNN lê [Batch, Canais(Sensores), Tempo]
        # Então precisamos transpor as dimensões:
        x = x.transpose(1, 2)
        x = self.conv_layer(x)
        x = self.flatten(x)
        previsao_rul = self.fc_layer(x)
        return previsao_rul

n_sensores = X_treino.shape[1]
modelo = CNN1DPredictor(n_features=n_sensores).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)

EPOCHS = 10
print("\nIniciando Treinamento da CNN 1D...")
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

print(f"Treinamento CNN concluído em {(time.time()-start)/60:.2f} minutos!")
torch.save(modelo.state_dict(), 'modelo_cnn.pth')

# === GRÁFICO DA CURVA DE APRENDIZADO ===
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS+1), historico_treino, label='Erro de Treino', color='blue', marker='o')
plt.plot(range(1, EPOCHS+1), historico_teste, label='Erro de Teste (Validação)', color='red', marker='s')
plt.xlabel('Épocas')
plt.ylabel('RMSE (Ciclos de Erro)')
plt.title('Curva de Aprendizado CNN 1D (Detecção de Overfitting)')
plt.legend()
plt.grid(True)
plt.savefig('grafico_cnn_learning_curve.png', bbox_inches='tight')
print("Gráfico da Curva de Aprendizado (Loss Curve) gerado com sucesso!")
print("\nTestando o modelo CNN...")
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

print("\n=== RESULTADOS DA CNN 1D ===")
print(f"Erro Absoluto Médio (MAE): {erro_mae:.2f} ciclos")
print(f"Erro Quadrático Médio (RMSE): {erro_rmse:.2f} ciclos")
print("=============================")

plt.figure(figsize=(10, 5))
plt.plot(Y_real[:500], label='RUL Real', color='blue', linewidth=2)
plt.plot(Y_previsto[:500], label='RUL Previsto (CNN)', color='green', linestyle='dashed', linewidth=2)
plt.xlabel('Janelas Temporais')
plt.ylabel('RUL')
plt.title('Trajetória de Degradação (Filtros da CNN)')
plt.legend()
plt.grid(True)
plt.savefig('grafico_cnn_trajetoria.png', bbox_inches='tight')
print("Gráfico salvo!")
