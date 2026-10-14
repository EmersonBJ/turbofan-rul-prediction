import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configuração visual profissional (estilo técnico de artigo científico)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.titleweight'] = 'bold'

# 1. Dados dos Modelos (Rankeados por RMSE de Teste)
modelos = [
    'XGBoost Otimizado',
    'Rede Suprema (Bi-LSTM)',
    'LSTM Padrao',
    'Hibrido CNN-LSTM',
    'CNN 1D Pura',
    'SVR (Kernel RBF)',
    'Transformer Encoder',
    'Regressao Linear'
]

rmse_treino = np.array([3.80, 3.65, 1.78, 1.53, 4.78, 5.75, 4.10, 15.60])
rmse_teste  = np.array([4.48, 6.39, 8.04, 7.62, 8.11, 7.09, 7.62, 15.97])
gap_overfitting = rmse_teste - rmse_treino

# Curvas de Aprendizado (Evolucao Epoca a Epoca)
epocas = np.arange(1, 11)
# LSTM: Epoca 1 a 10
lstm_teste = [7.66, 6.48, 7.81, 7.49, 7.50, 7.48, 7.47, 7.60, 7.80, 8.04]
lstm_treino = [6.77, 2.91, 3.26, 2.57, 2.30, 2.10, 1.98, 1.89, 1.82, 1.78]

# CNN 1D: Epoca 1 a 10
cnn_teste = [7.36, 7.23, 6.96, 7.10, 6.99, 7.30, 7.84, 7.95, 8.05, 8.11]
cnn_treino = [6.86, 5.95, 5.65, 5.40, 5.16, 5.08, 4.99, 4.90, 4.82, 4.78]

# Hibrido CNN-LSTM:
hibrido_teste = [6.83, 7.61, 7.65, 7.70, 7.72, 7.70, 7.68, 7.72, 7.76, 7.62]
hibrido_treino = [4.75, 2.66, 2.35, 2.10, 1.95, 1.82, 1.70, 1.62, 1.55, 1.53]

# Suprema (Bi-LSTM): com intervencao de Early Stopping na Epoca 2/5
suprema_epocas = [1, 2, 3, 4, 5]
suprema_teste = [7.98, 6.39, 6.45, 7.33, 7.32]
suprema_treino = [3.12, 3.65, 3.63, 3.02, 1.76]

# Criar Painel 2x2 de Alta Resolucao
fig, axs = plt.subplots(2, 2, figsize=(17, 13), dpi=300)
fig.suptitle('ANALISE INTEGRADA: DESEMPENHO, OVERFITTING E UNDERFITTING\nPredicao de RUL em Turbofans Aeronauticos (NASA N-CMAPSS DS02)', y=0.98)

# -------------------------------------------------------------
# PAINEL 1: Barras Comparativas de RMSE (Treino vs Teste e Gap)
# -------------------------------------------------------------
ax1 = axs[0, 0]
y_pos = np.arange(len(modelos))
bar_width = 0.35

rects1 = ax1.barh(y_pos - bar_width/2, rmse_treino, bar_width, label='RMSE Treino', color='#2b5c8f', alpha=0.9)
rects2 = ax1.barh(y_pos + bar_width/2, rmse_teste, bar_width, label='RMSE Teste Cego', color='#d95f02', alpha=0.9)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(modelos)
ax1.invert_yaxis()  # O melhor fica no topo
ax1.set_xlabel('Erro Quadratico Medio - RMSE (Ciclos)')
ax1.set_title('(A) Desempenho Relativo e Gap de Generalizacao (Treino vs. Teste)')
ax1.grid(axis='x', linestyle='--', alpha=0.5)
ax1.legend(loc='lower right', framealpha=0.9)

# Anotacoes de Gap de Overfitting
for i in range(len(modelos)):
    gap = gap_overfitting[i]
    val_max = max(rmse_teste[i], rmse_treino[i])
    if gap > 4.0:
        cor_gap = '#b30000'
        rotulo_diag = f'Delta: +{gap:.2f} (Overfitting Critico)'
    elif gap < 1.0 and rmse_teste[i] > 10.0:
        cor_gap = '#666666'
        rotulo_diag = f'Delta: +{gap:.2f} (Underfitting)'
    elif gap < 1.0:
        cor_gap = '#006d2c'
        rotulo_diag = f'Delta: +{gap:.2f} (Excelente)'
    else:
        cor_gap = '#fe9929'
        rotulo_diag = f'Delta: +{gap:.2f} (Moderado)'
        
    ax1.text(val_max + 0.3, y_pos[i], rotulo_diag, va='center', fontsize=8, fontweight='bold', color=cor_gap)

ax1.set_xlim(0, 25)
ax1.legend(loc='center right', framealpha=0.9)
# -------------------------------------------------------------
# PAINEL 2: Diagrama de Dispersao de Vies-Variancia (Trade-Off)
# -------------------------------------------------------------
ax2 = axs[0, 1]

# Definir Zonas Conceituais de Fundo
ax2.axvspan(-0.5, 2.0, ymin=0, ymax=0.45, color='#e5f5e0', alpha=0.6, label='Zona Otima (Equilibrio de Pareto)')
ax2.axvspan(3.0, 7.5, ymin=0.25, ymax=1.0, color='#fee0d2', alpha=0.6, label='Zona de Overfitting Severo (Alta Variancia)')
ax2.axvspan(-0.5, 2.0, ymin=0.65, ymax=1.0, color='#f0f0f0', alpha=0.7, label='Zona de Underfitting (Alto Vies)')

# Mapear cores dos pontos
cores_pontos = ['#006d2c', '#08519c', '#a50f15', '#cb181d', '#fe9929', '#54278f', '#756bb1', '#525252']
marcadores = ['*', 'o', 's', '^', 'D', 'v', 'p', 'X']
tamanhos = [220, 140, 130, 130, 120, 120, 120, 140]

for i in range(len(modelos)):
    ax2.scatter(gap_overfitting[i], rmse_teste[i], s=tamanhos[i], color=cores_pontos[i], marker=marcadores[i], edgecolors='black', linewidth=1.2, zorder=5)

# Rotulagem precisa dos pontos
deslocamentos = {
    'XGBoost Otimizado': (0.15, -0.4),
    'Rede Suprema (Bi-LSTM)': (-0.2, 0.45),
    'LSTM Padrao': (0.15, 0.25),
    'Hibrido CNN-LSTM': (-1.8, -0.55),
    'CNN 1D Pura': (0.15, 0.3),
    'SVR (Kernel RBF)': (0.15, -0.35),
    'Transformer Encoder': (-1.7, 0.35),
    'Regressao Linear': (0.15, 0.2)
}

for i, nome in enumerate(modelos):
    dx, dy = deslocamentos[nome]
    ax2.annotate(nome, (gap_overfitting[i], rmse_teste[i]), 
                 textcoords="offset points", xytext=(dx*20, dy*20), 
                 fontsize=8.5, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=cores_pontos[i], lw=1, alpha=0.85),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color=cores_pontos[i], lw=1),
                 zorder=6)

ax2.set_xlabel('Grau de Overfitting (Generalization Gap = RMSE Teste - RMSE Treino)')
ax2.set_ylabel('Erro de Teste Cego - RMSE (Ciclos)')
ax2.set_title('(B) Mapa de Diagnostico de Vies-Variancia (Desempenho vs. Overfitting)')
ax2.set_xlim(-0.2, 7.5)
ax2.set_ylim(3, 17.5)
ax2.grid(True, linestyle='--', alpha=0.5)

# Textos explicativos dentro das zonas
ax2.text(0.1, 4.0, 'EQUILIBRIO OTIMO\n(Baixo Erro + Baixo Gap)', fontsize=8, fontweight='bold', color='#006d2c', bbox=dict(boxstyle="square", fc="#e5f5e0", ec="#006d2c", alpha=0.8))
ax2.text(0.1, 14.2, 'UNDERFITTING\n(Modelo Linear Simples)', fontsize=8, fontweight='bold', color='#525252', bbox=dict(boxstyle="square", fc="#f0f0f0", ec="#525252", alpha=0.8))
ax2.text(4.2, 9.5, 'OVERFITTING CRITICO\n(LSTMs Memorizando Ruido)', fontsize=8, fontweight='bold', color='#a50f15', bbox=dict(boxstyle="square", fc="#fee0d2", ec="#a50f15", alpha=0.8))

# -------------------------------------------------------------
# PAINEL 3: Curvas de Aprendizado e Dinamica Temporal de Ruptura
# -------------------------------------------------------------
ax3 = axs[1, 0]

ax3.plot(epocas, lstm_teste, 'r-o', linewidth=2, label='LSTM Padrao (Teste)', zorder=4)
ax3.plot(epocas, lstm_treino, 'r--', linewidth=1.5, alpha=0.7, label='LSTM Padrao (Treino)', zorder=3)

ax3.plot(epocas, cnn_teste, color='#fe9929', marker='D', linewidth=2, label='CNN 1D (Teste)', zorder=4)
ax3.plot(epocas, cnn_treino, color='#fe9929', linestyle='--', linewidth=1.5, alpha=0.7, label='CNN 1D (Treino)', zorder=3)

ax3.plot(epocas, hibrido_teste, color='#cb181d', marker='^', linewidth=2, label='Hibrido CNN-LSTM (Teste)', zorder=4)

ax3.plot(suprema_epocas, suprema_teste, color='#08519c', marker='s', linewidth=2.5, label='Rede Suprema (Teste)', zorder=5)
ax3.plot(suprema_epocas, suprema_treino, color='#08519c', linestyle='--', linewidth=1.5, alpha=0.7, label='Rede Suprema (Treino)', zorder=3)

# Linhas de referencia estaticas (XGBoost e Baseline Linear)
ax3.axhline(4.48, color='#006d2c', linestyle='-', linewidth=2.5, label='XGBoost Otimizado (Patamar 4.48)', zorder=6)
ax3.axhline(15.97, color='#525252', linestyle=':', linewidth=2, label='Regressao Linear (Baseline 15.97)', zorder=2)

# Destaque de Early Stopping na Rede Suprema
ax3.scatter(2, 6.39, s=200, facecolors='none', edgecolors='#08519c', linewidth=2.5, zorder=7)
ax3.annotate('Pico da Suprema (Epoca 2)\nRMSE: 6.39 (Early Stop)', (2, 6.39), xytext=(2.3, 5.2),
             arrowprops=dict(facecolor='#08519c', shrink=0.08, width=1.5, headwidth=6),
             fontsize=8.5, fontweight='bold', color='#08519c', bbox=dict(boxstyle="round", fc="white", ec="#08519c", lw=1))

# Destaque do inicio da divergencia de Overfitting na LSTM
ax3.annotate('Bifurcacao Catastrofica (Overfitting)\nTreino despenca -> Teste explode', (3, 7.81), xytext=(3.3, 8.8),
             arrowprops=dict(facecolor='#a50f15', shrink=0.08, width=1.5, headwidth=6),
             fontsize=8.5, fontweight='bold', color='#a50f15', bbox=dict(boxstyle="round", fc="white", ec="#a50f15", lw=1))

ax3.set_xlabel('Epoca de Treinamento')
ax3.set_ylabel('RMSE (Ciclos)')
ax3.set_title('(C) Dinamica Temporal de Aprendizado e Divergencia de Generalizacao')
ax3.set_xticks(epocas)
ax3.set_ylim(1.0, 17.0)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', ncol=2, fontsize=7.5, framealpha=0.9)

# -------------------------------------------------------------
# PAINEL 4: Matriz Comparativa Estruturada de Diagnostico
# -------------------------------------------------------------
ax4 = axs[1, 1]
ax4.axis('off')

# Construcao da Tabela Sintetica de Diagnostico Tecnico
col_labels = ['Modelo / Arquitetura', 'RMSE Treino', 'RMSE Teste', 'Gap (Delta)', 'Diagnostico Tecnico']
table_data = [
    ['XGBoost Otimizado', '3.80', '4.48', '+0.68', 'Otimo: Generalizacao Superior'],
    ['Rede Suprema (Bi-LSTM)', '3.65', '6.39', '+2.74', 'Equilibrio via Early Stopping'],
    ['LSTM Padrao', '1.78', '8.04', '+6.26', 'Overfitting Severo (Decoreba)'],
    ['Hibrido CNN-LSTM', '1.53', '7.62', '+6.09', 'Overfitting Critico (Alta Var)'],
    ['CNN 1D Pura', '4.78', '8.11', '+3.33', 'Overfitting Moderado'],
    ['SVR (Kernel RBF)', '5.75', '7.09', '+1.34', 'Subotimo por Amostragem'],
    ['Transformer Encoder', '4.10', '7.62', '+3.52', 'Instabilidade perante Ruido'],
    ['Regressao Linear', '15.60', '15.97', '+0.37', 'Underfitting Severo (Alto Vies)']
]

cell_colors = [
    ['#e5f5e0', '#e5f5e0', '#e5f5e0', '#e5f5e0', '#e5f5e0'],  # Verde (XGBoost)
    ['#deebf7', '#deebf7', '#deebf7', '#deebf7', '#deebf7'],  # Azul (Suprema)
    ['#fee0d2', '#fee0d2', '#fee0d2', '#fee0d2', '#fee0d2'],  # Vermelho (LSTM)
    ['#fee0d2', '#fee0d2', '#fee0d2', '#fee0d2', '#fee0d2'],  # Vermelho (Hibrido)
    ['#fff7bc', '#fff7bc', '#fff7bc', '#fff7bc', '#fff7bc'],  # Amarelo (CNN 1D)
    ['#f7f7f7', '#f7f7f7', '#f7f7f7', '#f7f7f7', '#f7f7f7'],  # Cinza (SVR)
    ['#fff7bc', '#fff7bc', '#fff7bc', '#fff7bc', '#fff7bc'],  # Amarelo (Transformer)
    ['#f0f0f0', '#f0f0f0', '#f0f0f0', '#f0f0f0', '#f0f0f0']   # Cinza escuro (Linear)
]

table = ax4.table(cellText=table_data, colLabels=col_labels, cellColours=cell_colors,
                  colWidths=[0.30, 0.14, 0.14, 0.14, 0.38],
                  loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(8.2)
table.scale(1.0, 1.9)

# Estilizar o cabecalho da tabela
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#252525')
    else:
        cell.set_edgecolor('#cccccc')

ax4.set_title('(D) Matriz Sintetica de Diagnostico e Metricas de Confiabilidade', pad=15)

# Salvar e fechar figura
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_path = r'c:\Users\User\Downloads\ML\grafico_comparativo_desempenho_overfitting.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Grafico salvo com sucesso em: {output_path}")
