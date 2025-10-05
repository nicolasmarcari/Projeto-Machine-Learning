import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --------------------------
# 1. Baixar dados do Yahoo Finance
# --------------------------
ticker = 'PETR4.SA'  # Troque por 'PETR4.SA' ou outro código da B3 se quiser
df = yf.download(ticker, start='2011-01-01', end='2025-01-01')
df = df[['Close']]
df.dropna(inplace=True)

# --------------------------
# 2. Normalizar os dados
# --------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# --------------------------
# 3. Criar sequências (janelas)
# --------------------------
lookback = 60

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, lookback)

# Dividir em treino e teste
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Converter para tensores do PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

# --------------------------
# 4. Dataset e DataLoader
# --------------------------
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=True)

# --------------------------
# 5. Modelo LSTM
# --------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 6. Treinamento
# --------------------------
epochs = 20
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')

# --------------------------
# 7. Previsões
# --------------------------
model.eval()
with torch.no_grad():
    train_pred = model(X_train).numpy()
    test_pred = model(X_test).numpy()

# --------------------------
# 8. Inverter normalização
# --------------------------
train_pred_inv = scaler.inverse_transform(train_pred.reshape(-1, 1))
test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1))
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# --------------------------
# 9. Construir índices de datas corretos
# --------------------------
train_dates = df.index[lookback : lookback + len(y_train)]
test_dates = df.index[lookback + len(y_train) : lookback + len(y_train) + len(y_test)]

# --------------------------
# 10. Plotar resultados
# --------------------------
plt.figure(figsize=(14,6))
plt.plot(train_dates, y_train_inv, label='Preço Real - Treino')
plt.plot(train_dates, train_pred_inv, label='Previsão - Treino')
plt.plot(test_dates, y_test_inv, label='Preço Real - Teste')
plt.plot(test_dates, test_pred_inv, label='Previsão - Teste')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento')
plt.title(f'Previsão do Preço de Ações - {ticker}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
