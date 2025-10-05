import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------- CONFIG ----------
# tickers_treinamento = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'B3SA3.SA', 'BBAS3.SA']
# tickers_treinamento = [ 'PETR4.SA' ]
# ticker_previsao = 'WEGE3.SA'


tickers_treinamento = ['ITSA4.SA', 'ITSA3.SA']
ticker_previsao = 'ITSA4.SA'

start_date = '2015-01-01'
end_train = '2023-12-31'
start_forecast = '2024-01-01'
end_forecast = '2024-12-31'
pandemic_start = '2020-03-01'
pandemic_end = '2021-12-31'
window_size = 30
batch_size = 64
epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- FUNÇÕES AUXILIARES ----------
def add_technical_indicators(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df = df.dropna()
    return df

def create_sequences(data, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
        labels.append(data[i+window_size])  # prever todas as features no próximo passo
    return np.array(sequences), np.array(labels)

class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=100, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, input_size)  # saída multivariada

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# ---------- ETAPA 1: TREINAMENTO ----------
print("🔁 Preparando dados para treino...")

all_train_X = []
all_train_y = []

for ticker in tickers_treinamento:
    df = yf.download(ticker, start=start_date, end=end_train, auto_adjust=True)
    df = df[~((df.index >= pandemic_start) & (df.index <= pandemic_end))]
    df = add_technical_indicators(df)
    
    features = df[['Open','High','Low','Close','Volume','SMA_10','SMA_20']].values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    X, y = create_sequences(features_scaled, window_size)
    all_train_X.append(X)
    all_train_y.append(y)

# Concatenar dados
X_train = np.concatenate(all_train_X, axis=0)
y_train = np.concatenate(all_train_y, axis=0)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Modelo
model = LSTMModel(input_size=7, hidden_size=100, num_layers=2, dropout=0.2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Treinar
print("🚀 Treinando modelo...")
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.6f}")
print("✅ Treinamento finalizado.\n")

# ---------- ETAPA 2: PREVISÃO ----------
print(f"🔍 Baixando dados para {ticker_previsao} e preparando previsão para 2024...")

df = yf.download(ticker_previsao, start=start_date, end=end_train, auto_adjust=True)
df = df[~((df.index >= pandemic_start) & (df.index <= pandemic_end))]
df = add_technical_indicators(df)
historical_data = df.copy()

features = df[['Open','High','Low','Close','Volume','SMA_10','SMA_20']].values
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

dias_uteis_2024 = pd.date_range(start=start_forecast, end=end_forecast, freq='B')

last_window = features_scaled[-window_size:].copy()

model.eval()
forecast_scaled_all = []

window = last_window.copy()
for _ in range(len(dias_uteis_2024)):
    seq = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(seq).cpu().numpy().flatten()
    # Atualiza a janela com a previsão multivariada
    window = np.vstack([window[1:], pred])
    forecast_scaled_all.append(pred)

forecast_scaled_all = np.array(forecast_scaled_all)
forecast_all = scaler.inverse_transform(forecast_scaled_all)
forecast_close = forecast_all[:, 3]  # Close previsto

# ---------- ETAPA 3: VALORES REAIS ----------
real_df = yf.download(ticker_previsao, start=start_forecast, end=end_forecast, auto_adjust=True)
real_close = real_df['Close']

# ---------- ETAPA 4: MÉTRICAS ----------
min_len = min(len(real_close), len(forecast_close))

rmse = np.sqrt(mean_squared_error(real_close[:min_len], forecast_close[:min_len]))
mae = mean_absolute_error(real_close[:min_len], forecast_close[:min_len])
print(f"📊 RMSE: {rmse:.4f}")
print(f"📊 MAE: {mae:.4f}\n")

# ---------- ETAPA 5: GRÁFICO CONTÍNUO COMPARATIVO ----------
previsao_completa_close = pd.concat([
    historical_data['Close'],
    pd.Series(forecast_close[:min_len], index=dias_uteis_2024[:min_len])
])

real_completo_close = pd.concat([
    historical_data['Close'],
    real_close[:min_len]
])

plt.figure(figsize=(15,6))
plt.plot(real_completo_close.index, real_completo_close.values, label='Valor Real', color='green')
plt.plot(previsao_completa_close.index, previsao_completa_close.values, label='Valor Preditivo', color='orange', linestyle='--')

plt.axvline(pd.to_datetime(start_forecast), color='red', linestyle=':', linewidth=2, label='Início da Predição')

plt.title(f'{ticker_previsao} - Comparativo: Histórico + Previsão 2024 vs Real')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
