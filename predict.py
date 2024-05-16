import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Dados fornecidos
metals = {
    'months': [
        'August 2021', 'September 2021', 'October 2021', 'November 2021', 'December 2021', 'January 2022', 'February 2022',
        'March 2022', 'April 2022', 'May 2022', 'June 2022', 'July 2022', 'August 2022', 'September 2022', 'October 2022', 
        'November 2022', 'December 2022', 'January 2023', 'February 2023', 'March 2023', 'April 2023', 'May 2023', 
        'June 2023', 'July 2023', 'August 2023', 'September 2023', 'October 2023', 'November 2023', 'December 2023', 
        'January 2024', 'February 2024'
    ],
    'Copper': [
        8.240, 8.340, 8.123, 8.654, 8.504, 8.543, 8.123, 8.456, 8.785, 8.343, 8.123, 8.253, 8.463, 7.123, 8.123, 8.954, 
        8.430, 8.341, 8.864, 8.034, 8.690, 8.594, 8.470, 8.475, 8.3506, 8.2695, 7.9388, 8.173, 8.3932, 8.3427, 8.3094
    ],
    'Aluminum': [
        2.134, 2.314, 2.235, 2.235, 2.797, 2.796, 2.567, 2.097, 2.567, 2.789, 2.456, 2.786, 2.456, 2.789, 2.567, 2.645, 
        2.373, 2.345, 2.945, 2.234, 2.356, 2.175, 2.934, 2.923, 2.1334, 2.1769, 2.1918, 2.2016, 2.1737, 2.1936, 2.1819
    ],
    'BCB_Dollar': [
        5.349, 5.539, 5.235, 5.956, 5.845, 5.349, 5.023, 5.243, 5.124, 5.436, 5.253, 5.129, 5.964, 5.124, 5.346, 5.629, 
        5.123, 5.946, 5.358, 5.230, 5.469, 5.234, 5.366, 5.345, 4.9034, 4.9364, 5.0642, 4.8977, 4.8998, 4.9138, 4.9638
    ]
}

df = pd.DataFrame(metals)
df['months'] = pd.to_datetime(df['months'], format='%B %Y')
df.set_index('months', inplace=True)

def criar_janelas(df, janela_tamanho=6):
    X, y = [], []
    for i in range(len(df) - janela_tamanho):
        X.append(df.iloc[i:(i + janela_tamanho)].values)
        y.append(df.iloc[i + janela_tamanho].values)
    return np.array(X), np.array(y)

janela_tamanho = 12  # Aumentando o tamanho da janela
X, y = criar_janelas(df, janela_tamanho)

model = keras.Sequential([
    layers.Input(shape=(janela_tamanho, X.shape[2])),
    layers.LSTM(256, activation='relu', return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(128, activation='relu', return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(X.shape[2], activation='linear')
])

# Reduzindo a taxa de aprendizagem
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mse')

# Dividir os dados em conjuntos de treino e teste
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self, accuracy_threshold=0.95):
        super(CustomEarlyStopping, self).__init__()
        self.accuracy_threshold = accuracy_threshold

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(X_test)
        limiar = 0.10
        acuracias = []
        for i in range(y_test.shape[1]):
            dentro_limiar = np.abs((y_test[:, i] - y_pred[:, i]) / y_test[:, i]) < limiar
            acuracia = np.mean(dentro_limiar)
            acuracias.append(acuracia)
            print(f'Acurácia para {df.columns[i]}: {acuracia * 100:.2f}%')

        acuracia_media = np.mean(acuracias)
        print(f'Acurácia média: {acuracia_media * 100:.2f}%')

        if acuracia_media >= self.accuracy_threshold:
            print(f'Parando o treinamento na epoch {epoch + 1} pois a acurácia média está acima de {self.accuracy_threshold * 100:.2f}%')
            self.model.stop_training = True

early_stopping = CustomEarlyStopping(accuracy_threshold=0.95)

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Avaliar o modelo no conjunto de teste
loss = model.evaluate(X_test, y_test)
print(f'Loss (MSE) no conjunto de teste: {loss}')

# Selecionar os últimos dados para fazer a previsão
ultimo_periodo = df.iloc[-janela_tamanho:].values
ultimo_periodo = np.expand_dims(ultimo_periodo, axis =0)

# Fazer a previsão para o próximo mês
proxima_previsao = model.predict(ultimo_periodo)

# Converter a previsão para um DataFrame
colunas = ['Copper', 'Aluminum', 'BCB_Dollar']
proxima_previsao_df = pd.DataFrame(proxima_previsao, columns=colunas, index=[df.index[-1] + pd.DateOffset(months=1)])

print(proxima_previsao_df)

