import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st
import matplotlib.pyplot as plt

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
        8.240, 8.340, 8.123, 8.654, 8.504, 8.543, 8.350, 8.456, 8.785, 8.343, 8.345, 8.253, 8.463, 7.578, 8.478, 8.567, 
        8.430, 8.341, 8.678, 8.489, 8.690, 8.594, 8.470, 8.475, 8.3506, 8.2695, 7.9388, 8.173, 8.3932, 8.3427, 8.3094
    ],
    'Aluminum': [
        2.134, 2.314, 2.235, 2.235, 2.797, 2.796, 2.567, 2.097, 2.567, 2.789, 2.456, 2.786, 2.456, 2.789, 2.567, 2.645, 
        2.373, 2.345, 2.945, 2.234, 2.356, 2.175, 2.934, 2.923, 2.1334, 2.1769, 2.1918, 2.2016, 2.1737, 2.1936, 2.1819
    ],
    'BCB_Dollar': [
        5.349, 5.539, 5.235, 5.435, 5.755, 5.349, 5.023, 5.243, 5.124, 5.436, 5.253, 5.129, 5.765, 5.124, 5.346, 5.629, 
        5.123, 5.657, 5.358, 5.230, 5.469, 5.234, 5.366, 5.345, 4.9034, 4.9364, 5.0642, 4.8977, 4.8998, 4.9138, 4.9638
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

# Carregar o modelo
model = keras.models.load_model('LMETESTE.h5')

# Interface Streamlit
st.title("Previsão de Metais com Deep Learning")

# Mostrar DataFrame original
st.subheader("DataFrame Original")
st.write(df)

# Previsão para o próximo mês
if st.button("Prever próximo mês"):
    ultimo_periodo = df.iloc[-janela_tamanho:].values
    ultimo_periodo = np.expand_dims(ultimo_periodo, axis=0)
    proxima_previsao = model.predict(ultimo_periodo)

    # Converter a previsão para um DataFrame
    colunas = ['Copper', 'Aluminum', 'BCB_Dollar']
    proxima_previsao_df = pd.DataFrame(proxima_previsao, columns=colunas, index=[df.index[-1] + pd.DateOffset(months=1)])

    # Mostrar DataFrame com a previsão do próximo mês
    st.subheader("Próxima Previsão")
    st.write(proxima_previsao_df)

    # Gráfico
    st.subheader("Gráfico de Previsão")
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Copper'], label='Copper - Original')
    plt.plot(proxima_previsao_df.index, proxima_previsao_df['Copper'], label='Copper - Previsão')
    plt.xlabel("Data")
    plt.ylabel("Valor")
    plt.title("Previsão de Copper para o Próximo Mês")
    plt.legend()
    st.pyplot(plt)
