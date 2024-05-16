import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Normalization
from tensorflow.keras.optimizers import Adam

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

# Adicionar uma coluna de 'target' binária para a classificação binária
df['target'] = (df['Copper'] > 8.5).astype(int)

# Feature Engineering
df['Copper_Aluminum_Ratio'] = df['Copper'] / df['Aluminum']
df['Copper_BCB_Dollar_Ratio'] = df['Copper'] / df['BCB_Dollar']


dados_treino = df.sample(frac=0.75, random_state=1337)
dados_teste = df.drop(dados_treino.index)

print("Usando %d amostras para treino e %d para testar a acurácia do modelo"%(len(dados_treino), len(dados_teste)))

def dataframe_para_dataset(df):
    df = df.copy()
    labels = df.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    return ds

treino_ds = dataframe_para_dataset(dados_treino)
teste_ds = dataframe_para_dataset(dados_teste)

treino_ds = treino_ds.batch(32)
teste_ds = teste_ds.batch(32)

def normalizacao_variavel_numericas(variavel, nome, ds):
    normalizer = Normalization()
    variavel_ds = ds.map(lambda x, y: tf.expand_dims(x[nome], -1))
    normalizer.adapt(variavel_ds)
    variavel_normalizada = normalizer(variavel)
    return variavel_normalizada

# Inputs
copper = keras.Input(shape=(1,), name="Copper")
aluminum = keras.Input(shape=(1,), name="Aluminum")
bcb_dollar = keras.Input(shape=(1,), name="BCB_Dollar")
copper_aluminum_ratio = keras.Input(shape=(1,), name="Copper_Aluminum_Ratio")
copper_bcb_dollar_ratio = keras.Input(shape=(1,), name="Copper_BCB_Dollar_Ratio")

all_inputs = [copper, aluminum, bcb_dollar, copper_aluminum_ratio, copper_bcb_dollar_ratio]

# Normalization
copper_encoded = normalizacao_variavel_numericas(copper, "Copper", treino_ds)
aluminum_encoded = normalizacao_variavel_numericas(aluminum, "Aluminum", treino_ds)
bcb_dollar_encoded = normalizacao_variavel_numericas(bcb_dollar, "BCB_Dollar", treino_ds)
copper_aluminum_ratio_encoded = normalizacao_variavel_numericas(copper_aluminum_ratio, "Copper_Aluminum_Ratio", treino_ds)
copper_bcb_dollar_ratio_encoded = normalizacao_variavel_numericas(copper_bcb_dollar_ratio, "Copper_BCB_Dollar_Ratio", treino_ds)

all_features = layers.concatenate([copper_encoded, aluminum_encoded, bcb_dollar_encoded, copper_aluminum_ratio_encoded, copper_bcb_dollar_ratio_encoded])

# Model layers
x = layers.Dense(units=64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(all_features)
x = layers.Dropout(0.5)(x)
x = layers.Dense(units=32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
output = layers.Dense(units=1, activation="sigmoid")(x)

modelo = keras.Model(all_inputs, output)

modelo.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
modelo.fit(
    x=treino_ds,
    epochs=200,
    validation_data=teste_ds,
    callbacks=[early_stopping]
)
