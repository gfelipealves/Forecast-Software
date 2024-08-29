import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import resample
import streamlit as st
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_excel(r"Dados/FinalDataframeAuto.xlsx")

# Excluir a coluna Rank se existir
if 'Rank' in df.columns:
    del df['Rank']

# Preprocessamento dos dados
timefmt = "%d/%m/%Y"
df['Data'] = pd.to_datetime(df['Data'], format=timefmt)
df['Modelo'] = df['Modelo'].str.replace(' ', '').str.upper()

# Assuming listModelos is already populated
filteredData = df[['Data', 'Quantidade', 'Modelo']].query('Data >= "2024-01-01"')
listModelos = filteredData[['Modelo']].sort_values(by=['Modelo'], ascending=True).drop_duplicates(subset='Modelo')

# Convert the 'Modelo' column to a list
listModelos = listModelos['Modelo'].tolist()

# Build the Streamlit app
st.sidebar.title("Select the information of Assembler/Model of Vehicle")

# Create a dropdown list in Streamlit
selected_modelo = st.sidebar.selectbox("Select the Assembler/Model", listModelos)

# Display the selected value
st.sidebar.write(f"You selected: {selected_modelo}")

# Filtrar o modelo específico
analiseAuto = df[df['Modelo'] == selected_modelo][['Data', 'Quantidade']]

# Supondo que AnaliseAutoModelo seja o DataFrame original
analiseAuto = analiseAuto.set_index('Data')

# Converta o índice para o tipo DatetimeIndex
analiseAuto.index = pd.to_datetime(analiseAuto.index)

# Defina a frequência do índice como mensal
analiseAuto = analiseAuto.asfreq('MS')

# Identificar o período de 2010-01-01 a 2019-12-01
periodo_referencia = (analiseAuto.index >= '2022-01-01') & (analiseAuto.index <= '2024-05-01')

# Calcular a mediana para cada mês de janeiro de 2010 a dezembro de 2019
medianas_por_mes = analiseAuto.loc[periodo_referencia].groupby(analiseAuto.loc[periodo_referencia].index.month)['Quantidade'].median()

# Identificar o período de 2020-01-01 a 2021-12-01
periodo_substituicao = (analiseAuto.index >= '2020-01-01') & (analiseAuto.index <= '2021-12-01')

# Substituir os valores NaN pela mediana do respectivo mês
for mes in range(1, 13):
    mask = (analiseAuto.index.month == mes) & periodo_substituicao
    analiseAuto.loc[mask, 'Quantidade'] = medianas_por_mes[mes]
    #analiseAuto.loc[mask, 'Quantidade'] = analiseAuto.loc[mask, 'Quantidade'].fillna(medianas_por_mes)

# Resetar o índice do DataFrame
analiseAuto.reset_index(inplace=True)

# Converta a coluna 'Data' para datetime
analiseAuto['Data'] = pd.to_datetime(analiseAuto['Data'])

# Ordene o DataFrame por data
analiseAuto.sort_values(by='Data', inplace=True)

# Extraia características temporais da coluna 'Data'
analiseAuto['Ano'] = analiseAuto['Data'].dt.year
analiseAuto['Mes'] = analiseAuto['Data'].dt.month
analiseAuto['Dia'] = analiseAuto['Data'].dt.day

# Divida os dados em features (X) e target (y)
X = analiseAuto.drop(columns=['Quantidade', 'Data'])
y = analiseAuto['Quantidade']

# Substituir valores NaN em X e y com a mediana
X.fillna(round(X.median(), 0), inplace=True)
y.fillna(round(y.median(), 0), inplace=True)

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Parâmetros do modelo XGBoost
params = {
    'objective': 'reg:squarederror',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': 42
}

# Número de amostras de bootstrap
n_bootstraps = 100

# Matriz para armazenar as previsões de cada bootstrap
bootstrap_preds = np.zeros((n_bootstraps, len(X_test)))

# Executar bootstrap
for i in range(n_bootstraps):
    # Amostrar com substituição
    X_resample, y_resample = resample(X_train, y_train, random_state=i)
    
    # Treinar o modelo
    model = XGBRegressor(**params)
    model.fit(X_resample, y_resample)
    
    # Fazer previsões no conjunto de teste
    y_pred_test = model.predict(X_test)
    
    # Armazenar as previsões
    bootstrap_preds[i, :] = y_pred_test

# Calcular a média das previsões
pred_mean = np.mean(bootstrap_preds, axis=0)

# Calcular os intervalos de confiança (por exemplo, 95%)
ci_lower = np.percentile(bootstrap_preds, 2.5, axis=0)
ci_upper = np.percentile(bootstrap_preds, 97.5, axis=0)

# Adicionar as previsões ao DataFrame original para visualização
analiseAuto['Previsão'] = np.nan
analiseAuto.loc[X_test.index, 'Previsão'] = np.round(pred_mean, 0)

# Criar novas datas para os próximos 12 meses
last_date = analiseAuto['Data'].max()
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=24, freq='MS')

# Criar DataFrame para os próximos 12 meses
future_df = pd.DataFrame({'Data': future_dates})
future_df['Ano'] = future_df['Data'].dt.year
future_df['Mes'] = future_df['Data'].dt.month
future_df['Dia'] = future_df['Data'].dt.day

# Prever os próximos 24 meses
future_X = future_df.drop(columns=['Data'])
future_preds = model.predict(future_X)

# Arredondar as previsões futuras
future_df['Previsão'] = np.round(future_preds, 0)
future_df['Quantidade'] = np.nan  # Colocamos NaN para manter a estrutura

# Concatenar os DataFrames para visualizar todas as previsões
result_df = pd.concat([analiseAuto, future_df], ignore_index=True)

# Title on Streamlit
st.markdown(
    "<h1 style='font-size:15px;'>Data with the median values by month of the Dates 2022 to 2024 in Covid period - 2020 to 2021</h1>",
    unsafe_allow_html=True
)
# Create a figure and axis object
fig, ax = plt.subplots(figsize=(25, 12))

# Plot the data
ax.plot(result_df['Data'], result_df['Quantidade'], label='Real Data', color='blue')
ax.plot(result_df['Data'], result_df['Previsão'], label='Prevision', color='red')
ax.fill_between(analiseAuto['Data'].iloc[X_test.index], ci_lower, ci_upper, color='pink', alpha=0.8)
ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('Sales', fontsize=20)
ax.set_title('Prevision of Sales with XGBoost', fontsize=20)
ax.legend(fontsize=20)

# Display the plot in the Streamlit app
st.pyplot(fig)  # Passes the figure to Streamlit for rendering

# Calcular métricas de desempenho para o período testado
mae = mean_absolute_error(y_test, pred_mean)
mse = mean_squared_error(y_test, pred_mean)
rmse = np.sqrt(mse)

st.success(f'MAE: {round(mae, 0)}')
st.success(f'MSE: {round(mse, 0)}')
st.success(f'RMSE: {round(rmse, 0)}')

# Title on Streamlit
st.markdown(
    "<h1 style='font-size:15px;'>Database with the prevision</h1>",
    unsafe_allow_html=True
)

# Display the DataFrame in the Streamlit app
st.dataframe(result_df)

############################ Model without tge median substitution for 2020 to 2021 ##########################################

# Setting 'Data' as index already removes it as a column, so there's no need to access it as a column.
analiseAll = df[df['Modelo'] == selected_modelo][['Data', 'Quantidade']]

# Convert the 'Data' column to datetime
analiseAll['Data'] = pd.to_datetime(analiseAll['Data'])

# Set 'Data' as the index
analiseAll = analiseAll.set_index('Data')

# Sort the DataFrame by the date index
analiseAll.sort_values(by='Data', inplace=True)

# Set the frequency of the index to monthly
analiseAll = analiseAll.asfreq('MS')

# Extract year, month, and day from the index
analiseAll['Ano'] = analiseAll.index.year
analiseAll['Mes'] = analiseAll.index.month
analiseAll['Dia'] = analiseAll.index.day

# Divida os dados em features (X) e target (y)
X = analiseAll.drop(columns=['Quantidade'])
y = analiseAll['Quantidade']

# Substituir valores NaN em X e y com a mediana
X.fillna(round(X.median(), 0), inplace=True)
y.fillna(round(y.median(), 0), inplace=True)

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Parâmetros do modelo XGBoost
params = {
    'objective': 'reg:squarederror',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': 42
}

# Número de amostras de bootstrap
n_bootstraps = 100

# Matriz para armazenar as previsões de cada bootstrap
bootstrap_preds = np.zeros((n_bootstraps, len(X_test)))

# Executar bootstrap
for i in range(n_bootstraps):
    # Amostrar com substituição
    X_resample, y_resample = resample(X_train, y_train, random_state=i)
    
    # Treinar o modelo
    model = XGBRegressor(**params)
    model.fit(X_resample, y_resample)
    
    # Fazer previsões no conjunto de teste
    y_pred_test = model.predict(X_test)
    
    # Armazenar as previsões
    bootstrap_preds[i, :] = y_pred_test

# Calcular a média das previsões
pred_mean = np.mean(bootstrap_preds, axis=0)

# Calcular os intervalos de confiança (por exemplo, 95%)
ci_lower = np.percentile(bootstrap_preds, 2.5, axis=0)
ci_upper = np.percentile(bootstrap_preds, 97.5, axis=0)

# Adicionar as previsões ao DataFrame original para visualização
analiseAll['Previsão'] = np.nan
analiseAll.loc[X_test.index, 'Previsão'] = np.round(pred_mean, 0)

# Get the last date using the index instead of a column
last_date = analiseAll.index.max()

# Create new dates for the next 24 months
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=24, freq='MS')

# Create DataFrame for the next 24 months
future_df = pd.DataFrame({'Data': future_dates})
future_df['Ano'] = future_df['Data'].dt.year
future_df['Mes'] = future_df['Data'].dt.month
future_df['Dia'] = future_df['Data'].dt.day

# Predict the next 24 months
future_X = future_df.drop(columns=['Data'])
future_preds = model.predict(future_X)

# Round future predictions
future_df['Previsão'] = np.round(future_preds, 0)
future_df['Quantidade'] = np.nan  # Set NaN to maintain structure

# Concatenate DataFrames to visualize all predictions
result_df = pd.concat([analiseAll.reset_index(), future_df], ignore_index=True)

# Title on Streamlit
st.markdown(
    "<h1 style='font-size:15px;'>Data without the median values by month of the Dates 2022 to 2024 in Covid period - 2020 to 2021</h1>",
    unsafe_allow_html=True
)

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(25, 12))

# Plot the real data and predictions
ax.plot(result_df['Data'], result_df['Quantidade'], label='Real Data', color='blue')
ax.plot(result_df['Data'], result_df['Previsão'], label='Prevision', color='red')

# Use the index for the fill_between to get the confidence intervals
ax.fill_between(analiseAll.index[-len(X_test):], ci_lower, ci_upper, color='pink', alpha=0.8)
ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('Sales', fontsize=20)
ax.set_title('Prevision of Sales with XGBoost', fontsize=20)
ax.legend(fontsize=20)

# Display the plot in the Streamlit app
st.pyplot(fig)  # Passes the figure to Streamlit for rendering

# Calcular métricas de desempenho para o período testado
mae = mean_absolute_error(y_test, pred_mean)
mse = mean_squared_error(y_test, pred_mean)
rmse = np.sqrt(mse)

st.success(f'MAE: {round(mae, 0)}')
st.success(f'MSE: {round(mse, 0)}')
st.success(f'RMSE: {round(rmse, 0)}')

# Title on Streamlit
st.markdown(
    "<h1 style='font-size:15px;'>Database with the prevision</h1>",
    unsafe_allow_html=True
)

# Display the DataFrame in the Streamlit app
st.dataframe(result_df)
