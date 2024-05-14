import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

metals = {
    'months': ['August 2023', 'September 2023', 'October 2023', 'November 2023', 'December 2023', 'January 2024', 'February 2024'],
    'Copper': [8.3506, 8.2695, 7.9388, 8.173, 8.3932, 8.3427, 8.3094],
    'Aluminum': [2.1334, 2.1769, 2.1918, 2.2016, 2.1737, 2.1936, 2.1819],
    'BCB_Dollar': [4.9034, 4.9364, 5.0642, 4.8977, 4.8998, 4.9138, 4.9638]
}
#exibindo dados
df = pd.DataFrame(metals)

# Separar os dados em X (features) e y (target) para cada coluna
X_copper = df[['Aluminum', 'BCB_Dollar']].values[:-1]
y_copper = df['Copper'].shift(-1).values[:-1]

X_aluminum = df[['Copper', 'BCB_Dollar']].values[:-1]
y_aluminum = df['Aluminum'].shift(-1).values[:-1]

X_bcb = df[['Copper', 'Aluminum']].values[:-1]
y_bcb = df['BCB_Dollar'].shift(-1).values[:-1]

# Dividir os dados em conjuntos de treino e teste para cada coluna
X_train_copper, X_test_copper, y_train_copper, y_test_copper = train_test_split(X_copper, y_copper, test_size=0.2, random_state=42)
X_train_aluminum, X_test_aluminum, y_train_aluminum, y_test_aluminum = train_test_split(X_aluminum, y_aluminum, test_size=0.2, random_state=42)
X_train_bcb, X_test_bcb, y_train_bcb, y_test_bcb = train_test_split(X_bcb, y_bcb, test_size=0.2, random_state=42)

# Treinar o modelo de regressão linear para cada coluna
model_copper = LinearRegression()
model_aluminum = LinearRegression()
model_bcb = LinearRegression()

model_copper.fit(X_train_copper, y_train_copper)
model_aluminum.fit(X_train_aluminum, y_train_aluminum)
model_bcb.fit(X_train_bcb, y_train_bcb)

# Fazer previsões para cada coluna
prediction_copper = model_copper.predict([df[['Aluminum', 'BCB_Dollar']].iloc[-1].values])[0]
prediction_aluminum = model_aluminum.predict([df[['Copper', 'BCB_Dollar']].iloc[-1].values])[0]
prediction_bcb = model_bcb.predict([df[['Copper', 'Aluminum']].iloc[-1].values])[0]

# Criar DataFrames para as previsões
prediction_df = pd.DataFrame({
    'months': ['Prediction'],
    'Copper': [prediction_copper],
    'Aluminum': [prediction_aluminum],
    'BCB_Dollar': [prediction_bcb]
})

# Concatenar DataFrames de previsão com DataFrame original
full_df = pd.concat([df, prediction_df], ignore_index=True)

# Exibir as previsões mais recentes
st.write("Previsões para o próximo mês:")
st.write(f"Previsão do preço do Copper: {prediction_copper}")
st.write(f"Previsão do preço do Aluminum: {prediction_aluminum}")
st.write(f"Previsão do preço do BCB Dollar: {prediction_bcb}")

# Exibir tabela com dados e previsões
st.write("Tabela de dados e previsões:")
st.table(full_df)