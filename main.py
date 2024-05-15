from bayesian_predic__ import BayPredict
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import argparse
import json

parser = argparse.ArgumentParser(description='Programa para rodar a rede Bayesiana paralelamente')
parser.add_argument('local', type=str, help='local do arquivo de entrada')
args = parser.parse_args()

local = args.local

#INDEX=4 #para escolher o arquivo neste caso será ind_004.json


with open(local, 'r') as f:
    params = json.load(f)

N_YEARS = params['n_years']
NORM_TYPE = params['norm_type']
N_CAT = params['n_cat']
TARGET_VRIABLE = params['target_variable']
TOPOLOGY = params['topologia']
DATA_SOURCE = params['data_source']
STARTING_YEAR = params['starting_year']


pred = BayPredict(N_YEARS, NORM_TYPE, N_CAT, TARGET_VRIABLE, TOPOLOGY, DATA_SOURCE, STARTING_YEAR)


pred.create_observed_inference_file([]) #cria um arquivo de inferências padrão com todas as inferências de variáveis externas da rede bayesiana


#_ = pred.get_cat_limits() #Retornaum dicionário com as extremidades de cada categoria de cada variável

_ = pred.get_predictions('modelo_arquivo_4.json') #realiza uma previsão dado um arquivo de inferências o arquivo padrão gerado tem o nome 'modelo_arquivo_4.json'

df = pred.show_predictions(plot=True) #retorna um dataframe com os limites superior e inferior além do valor médio da previsão

df.index = df.index.strftime('%Y-%m-%d')

df.to_json(f'resultados_{local.split(".")[0]}.json')

mape = mean_absolute_percentage_error(df['Preco_L'], df['mean'])
mse = mean_squared_error(df['Preco_L'], df['mean'])
mae = mean_absolute_error(df['Preco_L'], df['mean'])



print(f'''
MAPE: {mape}
MAE: {mae}
SMSE: {mse**(1/2)}
''')