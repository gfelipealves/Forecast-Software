import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import resample

# Carregar os dados
AnaliseAuto = pd.read_excel(r"Dados/FinalDataframeAuto.xlsx")

modeloMontadora = 'GM/ONIX'
# Excluir a coluna Rank se existir
if 'Rank' in AnaliseAuto.columns:
    del AnaliseAuto['Rank']

# Preprocessamento dos dados
timefmt = "%d/%m/%Y"
AnaliseAuto['Data'] = pd.to_datetime(AnaliseAuto['Data'], format=timefmt)
AnaliseAuto['Modelo'] = AnaliseAuto['Modelo'].str.replace(' ', '').str.upper()

# Filtrar o modelo específico
AnaliseAutoModelo = AnaliseAuto[AnaliseAuto['Modelo'] == modeloMontadora ][['Data', 'Quantidade']]

# Supondo que AnaliseAutoModelo seja o DataFrame original
AnaliseAuto = AnaliseAutoModelo.set_index('Data')

# Converta o índice para o tipo DatetimeIndex
AnaliseAuto.index = pd.to_datetime(AnaliseAuto.index)

# Defina a frequência do índice como mensal
AnaliseAuto = AnaliseAuto.asfreq('MS')

# Identificar o período de 2010-01-01 a 2019-12-01
periodo_referencia = (AnaliseAuto.index >= '2022-01-01') & (AnaliseAuto.index <= '2024-05-01')

# Calcular a mediana para cada mês de janeiro de 2010 a dezembro de 2019
medianas_por_mes = AnaliseAuto.loc[periodo_referencia].groupby(AnaliseAuto.loc[periodo_referencia].index.month)['Quantidade'].median()

# Identificar o período de 2020-01-01 a 2021-12-01
periodo_substituicao = (AnaliseAuto.index >= '2020-01-01') & (AnaliseAuto.index <= '2021-12-01')

# Substituir os valores de janeiro de 2020 a dezembro de 2021 pela mediana correspondente
for mes in range(1, 13):
    mask = (AnaliseAuto.index.month == mes) & periodo_substituicao
    AnaliseAuto.loc[mask, 'Quantidade'] = medianas_por_mes[mes]

# Resetar o índice do DataFrame
AnaliseAuto.reset_index(inplace=True)
# Converta a coluna 'Data' para datetime
AnaliseAuto['Data'] = pd.to_datetime(AnaliseAuto['Data'])

# Ordene o DataFrame por data
AnaliseAuto.sort_values(by='Data', inplace=True)

# Extraia características temporais da coluna 'Data'
AnaliseAuto['Ano'] = AnaliseAuto['Data'].dt.year
AnaliseAuto['Mes'] = AnaliseAuto['Data'].dt.month
AnaliseAuto['Dia'] = AnaliseAuto['Data'].dt.day

# Divida os dados em features (X) e target (y)
X = AnaliseAuto.drop(columns=['Quantidade', 'Data'])
y = AnaliseAuto['Quantidade']

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
AnaliseAuto['Previsão'] = np.nan
AnaliseAuto.loc[X_test.index, 'Previsão'] = np.round(pred_mean, 0)

# Criar novas datas para os próximos 12 meses
last_date = AnaliseAuto['Data'].max()
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
result_df = pd.concat([AnaliseAuto, future_df], ignore_index=True)
'''
# Salvar o modelo treinado em um arquivo .pkl
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)
'''