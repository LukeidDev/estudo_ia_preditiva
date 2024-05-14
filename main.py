import streamlit as st

st.page_link(fr"pages\app.py", label="table", icon="🏠")
st.page_link(fr"pages\predict.py", label="Prediction", icon="1️⃣")




#graficos
'''
import pandas as pd
import matplotlib.pyplot as plt

# Dados
metals = {
    'months': ['August 2023', 'September 2023', 'October 2023', 'November 2023', 'December 2023', 'January 2024', 'February 2024'],
    'Copper': [8.3506, 8.2695, 7.9388, 8.173, 8.3932, 8.3427, 8.3094],
    'Aluminum': [2.1334, 2.1769, 2.1918, 2.2016, 2.1737, 2.1936, 2.1819],
    'BCB_Dollar': [4.9034, 4.9364, 5.0642, 4.8977, 4.8998, 4.9138, 4.9638]
}

# Converter para DataFrame
df = pd.DataFrame(metals)

# Converter 'months' para datetime e definir como índice
df['months'] = pd.to_datetime(df['months'], format='%B %Y')
df.set_index('months', inplace=True)

# Plotar os preços do cobre, alumínio e câmbio ao longo do tempo
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(df.index, df['Copper'], color='blue')
plt.title('Copper Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')

plt.subplot(3, 1, 2)
plt.plot(df.index, df['Aluminum'], color='green')
plt.title('Aluminum Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')

plt.subplot(3, 1, 3)
plt.plot(df.index, df['BCB_Dollar'], color='red')
plt.title('BCB Dollar Exchange Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Rate')

plt.tight_layout()
plt.show()
'''