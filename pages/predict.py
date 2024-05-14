import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

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

# Adicionar frequência mensal aos índices do DataFrame
df.index.freq = 'MS'

# Função para ajustar e prever com ARIMA
def fit_and_predict(df, column, steps=3):
    model = ARIMA(df[column], order=(1, 1, 0))
    model_fit = model.fit()
    predictions = model_fit.predict(start=1, end=len(df[column]) - 1, typ='levels')
    future_forecast = model_fit.forecast(steps=steps)
    return predictions, future_forecast

# Previsões para Cobre
predictions_copper, future_forecast_copper = fit_and_predict(df, 'Copper')
mse_copper = mean_squared_error(df['Copper'][1:], predictions_copper)
print('Erro quadrático médio (MSE) para o cobre:', mse_copper)

# Previsões para Alumínio
predictions_aluminum, future_forecast_aluminum = fit_and_predict(df, 'Aluminum')
mse_aluminum = mean_squared_error(df['Aluminum'][1:], predictions_aluminum)
print('Erro quadrático médio (MSE) para o alumínio:', mse_aluminum)

# Previsões para Taxa de Câmbio
predictions_bcb_dollar, future_forecast_bcb_dollar = fit_and_predict(df, 'BCB_Dollar')
mse_bcb_dollar = mean_squared_error(df['BCB_Dollar'][1:], predictions_bcb_dollar)
print('Erro quadrático médio (MSE) para a taxa de câmbio:', mse_bcb_dollar)

# Plotando previsões futuras
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=3, freq='MS')

# Cobre
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Copper'], label='Observed', color='blue')
plt.plot(df.index[1:], predictions_copper, label='Predicted', color='red')
plt.plot(future_dates, future_forecast_copper, label='Future Forecast', color='green', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('ARIMA Model - Copper Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Alumínio
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Aluminum'], label='Observed', color='green')
plt.plot(df.index[1:], predictions_aluminum, label='Predicted', color='red')
plt.plot(future_dates, future_forecast_aluminum, label='Future Forecast', color='blue', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('ARIMA Model - Aluminum Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Taxa de Câmbio
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['BCB_Dollar'], label='Observed', color='red')
plt.plot(df.index[1:], predictions_bcb_dollar, label='Predicted', color='blue')
plt.plot(future_dates, future_forecast_bcb_dollar, label='Future Forecast', color='green', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Rate')
plt.title('ARIMA Model - BCB Dollar Exchange Rate Prediction')
plt.legend()
plt.grid(True)
plt.show()
