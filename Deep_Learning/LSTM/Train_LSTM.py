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

print("=== Iniciando Pipeline LSTM (PyTorch - Otimizado para RAM) ===")

# 1. Hardware (Forçando CPU)
# Nota: Forçamos o uso da CPU ('cpu') porque a placa de vídeo relatou esgotamento 
# de memória VRAM (CUDA Memory allocation failure) no último teste.
device = torch.device('cpu')
print(f"Hardware selecionado para a Rede Neural: {device} (Modo de Segurança de Memória)")

# 2. Carregamento Direto em NumPy (Corta o consumo de RAM brutalmente)
caminho_h5 = r"C:\Users\User\Downloads\ML\Simulated traning data\N CMAPSS\17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2\17. Turbofan Engine Degradation Simulation Data Set 2\data_set\data_set\N-CMAPSS_DS01-005.h5"
arquivo_h5 = h5py.File(caminho_h5, 'r')

def extrair_dados_numpy(arquivo, prefixo='dev'):
    # Puxar como float32 impede o estouro de memória (corta pela metade o peso)
    A = np.array(arquivo[f'A_{prefixo}'], dtype=np.float32)
    T = np.array(arquivo[f'T_{prefixo}'], dtype=np.float32)
    W = np.array(arquivo[f'W_{prefixo}'], dtype=np.float32)
    Xs = np.array(arquivo[f'X_s_{prefixo}'], dtype=np.float32)
    Xv = np.array(arquivo[f'X_v_{prefixo}'], dtype=np.float32)
    Y = np.array(arquivo[f'Y_{prefixo}'], dtype=np.float32)
    
    # A primeira coluna de A é o ID do Motor (Unit)
    unit_array = A[:, 0]
    
    # Remove o Unit_ID das Features e empilha o resto
    features = np.hstack([A[:, 1:], T, W, Xs, Xv])
    
    return features, Y, unit_array

print("Extraindo matrizes puras do H5 (evitando Pandas para poupar Memória)...")
X_treino, Y_treino, units_treino = extrair_dados_numpy(arquivo_h5, 'dev')
X_teste, Y_teste, units_teste    = extrair_dados_numpy(arquivo_h5, 'test')

# Limpa o H5 da memória ram
arquivo_h5.close()
gc.collect()

# 3. Normalização IN-PLACE (Sem cópias de memória para evitar MemoryError)
print("Aplicando Normalização Min-Max In-Place (Extremo baixo uso de RAM)...")
# Calculamos os mínimos e máximos apenas na base de treino
min_vals = X_treino.min(axis=0)
max_vals = X_treino.max(axis=0)
range_vals = max_vals - min_vals
range_vals[range_vals == 0] = 1.0  # Evita divisão por zero

# Modifica as matrizes DIRETAMENTE na memória (In-Place) sem criar cópias pesadas
X_treino -= min_vals
X_treino /= range_vals

X_teste -= min_vals
X_teste /= range_vals

# 4. A Mágica do 3D: PyTorch Lazy Loading
# Como gerar um array 3D com 4.9M de linhas custaria uns 25 GB de RAM, 
# nós criamos a janela de voo na hora exata em que a rede neural for "comer" o dado.
class CMAPSS_SequenceDataset(Dataset):
    def __init__(self, X, Y, units, window_size=30):
        self.X = X
        self.Y = Y
        self.window_size = window_size
        
        # Mapeia onde começam e terminam os motores para não cruzar dados
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
print(f"Mapeando janelas dinâmicas de {WINDOW_SIZE} voos (Modo Econômico de Memória)...")
dataset_treino = CMAPSS_SequenceDataset(X_treino, Y_treino, units_treino, window_size=WINDOW_SIZE)
dataset_teste  = CMAPSS_SequenceDataset(X_teste, Y_teste, units_teste, window_size=WINDOW_SIZE)

loader_treino = DataLoader(dataset_treino, batch_size=64, shuffle=True)
loader_teste  = DataLoader(dataset_teste, batch_size=64, shuffle=False)

# 5. A Arquitetura da Rede Neural LSTM (PyTorch)
class LSTMPredictor(nn.Module):
    def __init__(self, n_features, n_hidden=64, n_layers=2):
        super(LSTMPredictor, self).__init__()
        # A Camada de Memória (LSTM)
        self.lstm = nn.LSTM(
            input_size=n_features, 
            hidden_size=n_hidden, 
            num_layers=n_layers, 
            batch_first=True,    # Garante que as Amostras sejam a 1ª dimensão
            dropout=0.2          # Desliga 20% dos neurônios para evitar vício (Overfitting)
        )
        # Camada final para cuspir 1 único número (A previsão da Vida Útil)
        self.linear = nn.Linear(n_hidden, 1)
        
    def forward(self, x):
        # A LSTM retorna as memórias (lstm_out) e os estados ocultos
        lstm_out, _ = self.lstm(x)
        # Pegamos apenas a memória da ÚLTIMA leitura da janela temporal
        ultima_memoria = lstm_out[:, -1, :]
        previsao_rul = self.linear(ultima_memoria)
        return previsao_rul

n_sensores = X_treino.shape[1]
modelo = LSTMPredictor(n_features=n_sensores).to(device)

criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)

# 6. O Treinamento (Épocas com Avaliação de Validação)
EPOCHS = 10
print("\nIniciando Treinamento Deep Learning...")
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
        loss = criterion(previsoes, batch_y)
        loss.backward()
        optimizer.step()
        
        loss_treino_acumulada += loss.item()
        
    rmse_treino = np.sqrt(loss_treino_acumulada / len(loader_treino))
    historico_treino.append(rmse_treino)
    
    # --- FASE DE VALIDAÇÃO (TESTE CEGO NO FINAL DE CADA ÉPOCA) ---
    modelo.eval()
    loss_teste_acumulada = 0
    with torch.no_grad():
        for batch_x, batch_y in loader_teste:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            previsoes = modelo(batch_x)
            loss_teste = criterion(previsoes, batch_y)
            loss_teste_acumulada += loss_teste.item()
            
    rmse_teste = np.sqrt(loss_teste_acumulada / len(loader_teste))
    historico_teste.append(rmse_teste)
    
    print(f"Época {epoch+1}/{EPOCHS} - RMSE Treino: {rmse_treino:.2f} | RMSE Teste: {rmse_teste:.2f}")

print(f"Treinamento concluído em {(time.time()-start)/60:.2f} minutos!")
torch.save(modelo.state_dict(), 'modelo_lstm.pth')

# === GERANDO O GRÁFICO DA CURVA DE APRENDIZADO (OVERFITTING) ===
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS+1), historico_treino, label='Erro de Treino', color='blue', marker='o')
plt.plot(range(1, EPOCHS+1), historico_teste, label='Erro de Teste (Validação)', color='red', marker='s')
plt.xlabel('Épocas')
plt.ylabel('RMSE (Ciclos de Erro)')
plt.title('Curva de Aprendizado LSTM (Detecção de Overfitting)')
plt.legend()
plt.grid(True)
plt.savefig('grafico_lstm_learning_curve.png', bbox_inches='tight')
print("Gráfico da Curva de Aprendizado (Loss Curve) gerado com sucesso!")

# 7. Validação Final e Gráfico de Trajetória
print("\nGerando previsões finais da trajetória...")
modelo.eval()
Y_previsto_lista = []
Y_real_lista = []

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

print("\n=== RESULTADOS DA LSTM ===")
print(f"Erro Absoluto Médio (MAE): {erro_mae:.2f} ciclos")
print(f"Erro Quadrático Médio (RMSE): {erro_rmse:.2f} ciclos")
print("===========================")

# Salvando a Trajetória
plt.figure(figsize=(10, 5))
plt.plot(Y_real[:500], label='RUL Real', color='blue', linewidth=2)
plt.plot(Y_previsto[:500], label='RUL Previsto (LSTM)', color='darkorange', linestyle='dashed', linewidth=2)
plt.xlabel('Janelas Temporais')
plt.ylabel('RUL')
plt.title('Trajetória de Degradação (Memória LSTM)')
plt.legend()
plt.grid(True)
plt.savefig('grafico_lstm_trajetoria.png', bbox_inches='tight')
print("Gráficos de trajetória salvos com sucesso!")
