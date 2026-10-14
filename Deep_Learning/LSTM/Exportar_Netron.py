import h5py
import numpy as np
import torch
import torch.nn as nn

print("=== Gerador de Raio-X da Rede Neural (ONNX para Netron) ===")

# 1. Definimos a Mesma Estrutura da Nossa LSTM
class LSTMPredictor(nn.Module):
    def __init__(self, n_features, n_hidden=64, n_layers=2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2
        )
        self.linear = nn.Linear(n_hidden, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        ultima_memoria = lstm_out[:, -1, :]
        previsao_rul = self.linear(ultima_memoria)
        return previsao_rul

# 2. Descobrimos o número de sensores apenas lendo a primeira linha do H5
caminho_h5 = r"C:\Users\User\Downloads\ML\Simulated traning data\N CMAPSS\17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2\17. Turbofan Engine Degradation Simulation Data Set 2\data_set\data_set\N-CMAPSS_DS01-005.h5"
with h5py.File(caminho_h5, 'r') as arquivo_h5:
    A = np.array(arquivo_h5['A_dev'][:1], dtype=np.float32)
    T = np.array(arquivo_h5['T_dev'][:1], dtype=np.float32)
    W = np.array(arquivo_h5['W_dev'][:1], dtype=np.float32)
    Xs = np.array(arquivo_h5['X_s_dev'][:1], dtype=np.float32)
    Xv = np.array(arquivo_h5['X_v_dev'][:1], dtype=np.float32)
    features = np.hstack([A[:, 1:], T, W, Xs, Xv])
    n_sensores = features.shape[1]

print(f"Número de sensores detectado: {n_sensores}")

# 3. Montamos a rede
modelo = LSTMPredictor(n_features=n_sensores)

# Carregamos os pesos que treinamos ontem (opcional, mas deixa a exportação perfeita)
try:
    modelo.load_state_dict(torch.load('modelo_lstm.pth', map_location='cpu'))
    print("Pesos do treinamento carregados com sucesso!")
except FileNotFoundError:
    print("Arquivo modelo_lstm.pth não encontrado. Exportando a estrutura pura...")

modelo.eval()

# 4. Criamos um "dado fantasma" de 1 lote e janela de 30 voos
# Formato: [Batch_size=1, Window=30, Features=n_sensores]
dummy_input = torch.randn(1, 30, n_sensores)

# 5. Exportamos o Raio-X!
caminho_saida = "RaioX_LSTM.onnx"
torch.onnx.export(
    modelo,                      # A Rede Neural
    dummy_input,                 # O Tensor de exemplo que entra nela
    caminho_saida,               # Onde salvar
    export_params=True,          # Salvar os pesos matemáticos
    opset_version=11,            # Versão Universal Padrão
    input_names=['Sensores_Entrada'],   # Nome visual para a Entrada
    output_names=['Ciclos_RUL_Saida'],  # Nome visual para a Saída
    dynamic_axes={'Sensores_Entrada': {0: 'batch_size'}, 'Ciclos_RUL_Saida': {0: 'batch_size'}}
)

print(f"\nMAGNÍFICO! O arquivo '{caminho_saida}' foi gerado.")
print("Agora basta arrastar esse arquivo para a tela do site: https://netron.app")
