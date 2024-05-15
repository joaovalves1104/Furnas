import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import DBNInference
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import json
from statsmodels.tsa.seasonal import seasonal_decompose
import networkx as kx
from scipy.stats import gaussian_kde

class norm_type:
    CLUSTER = 'cl'
    QUANTIS = 'qt'

def estimador(df: pd.DataFrame, atributo: str, categoria: int, n_projecao: int) -> tuple:

        dataframe = pd.DataFrame({'data': df[df[atributo+'_CAT'] == categoria][atributo]})

        density = gaussian_kde(dataframe['data'])
        x_vals = np.linspace(min(dataframe['data']), max(dataframe['data']), len(df[df[atributo+'_CAT'] == categoria][atributo]))
        density_vals = density(x_vals)

        valores_projetados = density.resample(n_projecao)[0]

        # #projetar real, sorteados, e função de probabilidada
        # plt.hist(valores_projetados, bins=20, density=True, alpha=0.5, label='Valores Projetados')
        # plt.hist(dataframe['data'], bins=20, density=True, alpha=0.5, label='Histograma dos Dados')
        # plt.plot(x_vals, density_vals, label='Função de Densidade de Probabilidade')

        # # Adicionar títulos e legendas
        # plt.title('Histograma e Função de Distribuição')
        # plt.xlabel('Dados')
        # plt.ylabel('Densidade')
        # plt.legend()

        # # Mostrar o gráfico
        # plt.show()

        return np.mean(valores_projetados) - np.std(valores_projetados), np.mean(valores_projetados) + np.std(valores_projetados), np.mean(valores_projetados), np.std(valores_projetados)

class BayPredict:
    

    def __init__(self, N_YEARS, NORM_TYPE, N_CAT, TARGET_VRIABLE, TOPOLOGY, DATA_SOURCE, STARTING_YEAR):
        self.N_YEARS = N_YEARS
        self.NORM_TYPE = NORM_TYPE
        self.N_CAT = N_CAT
        self.TARGET_VRIABLE = TARGET_VRIABLE
        self.TOPOLOGY = TOPOLOGY
        self.DATA_SOURCE = DATA_SOURCE
        self.STARTING_YEAR = STARTING_YEAR
        self.NUMERIC_VARI = None
        self.DYNAMIC_BAYESIAN_TOPOLOGY = None


        base_df = pd.read_parquet(DATA_SOURCE)
        deseasoned_base_df = pd.DataFrame(index = base_df.index)
        df_sazonal = pd.DataFrame()

        #EXTRAINDO TENDENCIA DOS DADOS
        for column in base_df.columns:
            deseasoned_base_df.loc[:, column] = seasonal_decompose(base_df[column], model='multiplicative').trend #if column == self.TARGET_VRIABLE else base_df[column]
        deseasoned_base_df = deseasoned_base_df.dropna()

        #EXTRAINDO A SAZONALIDADE
        df_sazonal[TARGET_VRIABLE] = seasonal_decompose(base_df[TARGET_VRIABLE], model='multiplicative').seasonal



        n_years = N_YEARS #Número de anos da previsão
        year = STARTING_YEAR
        values = df_sazonal[df_sazonal.index.year==year].values
        for i in range(n_years):
            index = pd.date_range(f'{year + 1 + i}-01-01', periods=12, freq='MS')
            df_sazonal = pd.concat([df_sazonal, pd.DataFrame(values.flatten(), index=index, columns=[TARGET_VRIABLE])])


        self.DESEASONED_DF = deseasoned_base_df
        self.BASE_DF = base_df
        self.SEASONALITY = df_sazonal

        self.generate_dynamic_topology()

        
    def simple_normalization(self, df:pd.DataFrame, return_params=False)->pd.DataFrame:
        if return_params==True:
            return df.abs().max()

        return df/df.abs().max()

    # def log_normalization(df:pd.DataFrame, return_params=False)->pd.DataFrame:
    #     # if return_params==True:
    #     #     
        
    #     return df.apply(np.log1p, axis=1)

    # def zscore_normalization(df:pd.DataFrame, return_params=False)->pd.DataFrame:
    #     # if return_params==True:
    #     #     

    #     return df.apply(zscore)




    #Discretização pelo método de clusterização
    #return_params=True é utilizado na função que posteriormente transforma os valores categóricos para numéricos
    def k_means_disc(self, array, n_bins_dict='', return_params=False):

        if n_bins_dict != '':
            n_clusters = n_bins_dict[array.name]

        n_clusters = self.N_CAT

        #array = array.dropna()

        est = KBinsDiscretizer(n_bins=n_clusters, encode='ordinal', strategy='kmeans', subsample=None)
        x = np.array(array).reshape(-1, 1)
        #print(x)
        est.fit(x)
        x_cat = est.transform(x).reshape(1, -1)[0]

        if return_params==True:
            cluster_edges = est.bin_edges_[0]


            boundaries = {}
            for i in range(len(cluster_edges[:-1])):
                boundaries.update({i : (cluster_edges[i], cluster_edges[i+1])})

            return boundaries

        return [int(value) for value in x_cat]




    #Discretização baseada em quantis, de forma que cada categoria possui o mesmo número de valores observados
    #quartis=True sempre divide o dataset em 4 categorias
    #quantile sizes indica o número de categorias desejadas
    #return_params=True é utilizado na função que posteriormente transforma os valores categóricos para numéricos
    def quantile_discretization_pipeline(self, df:pd.DataFrame,
                                        n_bins_dict='',
                                        quartis=False,
                                        return_params=False)->pd.DataFrame:

        df = df.copy()

        quantile_sizes = self.N_CAT
        #listas de variáveis as quais não serão aplicadas a transformação em variação

        # donottransform = []

        columns = df.columns

        # df[columns.drop(donottransform)] = df[columns.drop(donottransform)].diff()

        df = (df.diff()/df.shift(1)).dropna()
        
        #df = df.pipe(self.simple_normalization)
        
        bins_dict = {}
        if quartis == False:

            delta = 1/quantile_sizes

            for column in df.columns:
                if n_bins_dict != '':
                    quantile_sizes = n_bins_dict[column]

                bins = [df[column].quantile(0.00) - 0.00000001] + [df[column].quantile(delta*i) for i in range(1, quantile_sizes + 1)]
                bins_dict.update({column : bins})
                labels_ = [i for i in range(quantile_sizes)]


        else:

            for column in df.columns:

                bins = [df[column].quantile(0.00)- 0.00000001,
                        df[column].quantile(0.25),
                        df[column].quantile(0.50),
                        df[column].quantile(0.75),
                        df[column].quantile(1.00)]

                bins_dict.update({column : bins})

                labels_ = [0,1,2,3]


        if return_params==True:
            return bins_dict

        out_df = pd.DataFrame(columns=columns)
        for column in df.columns:
            out_df[column] = pd.cut(df[column], bins=bins_dict[column],labels=labels_)

        return out_df
    

    #Função utilizada para retornar os parametros necessários na transformação das categorias para valores numéricos
    #Aqui são usadas as funções com return_params=True
    def get_norm_cat_params(self, df:pd.DataFrame,  n_bins_dict='', quartis=False):
        
        normalization_type = self.NORM_TYPE
        
        if normalization_type == norm_type.CLUSTER:
            #Retorna dicionário com os valores base da normalização (em um dataframe) e os limites de cada categoria (em um dataframe)
            # donottransform = []
            columns = df.columns
            # df[columns.drop(donottransform)] = df[columns.drop(donottransform)].diff()

            df = (df.diff()/df.shift(1)).dropna()

            return {'base_values': df.pipe(self.simple_normalization, return_params=True), 'boundaries': df.apply(self.k_means_disc, args=(n_bins_dict, True))}

        elif normalization_type == norm_type.QUANTIS:

            limits_dict = df.pipe(self.quantile_discretization_pipeline, return_params=True, n_bins_dict=n_bins_dict, quartis=quartis)
            boundaries_dict = {}
            for key in limits_dict.keys():
                individual_boundaries_dict = {}
                bins_list = limits_dict[key]
                for i in range(len(bins_list) - 1):
                    individual_boundaries_dict.update({i: (bins_list[i], bins_list[i+1])})
                boundaries_dict.update({key: individual_boundaries_dict})
            #Retorna dicionário com os valores base da normalização (em um dataframe) e os limites de cada categoria (em um dataframe)
            return {'base_values': df.diff().pipe(self.simple_normalization, return_params=True), 'boundaries': boundaries_dict}
        

    #Função que categoriza a base de dados
    #São criadas duas colunas para cada variável (nos tempos 0 e 1)
    #A coluna no tempo 1 é uma cópia da coluna no tempo 0 deslocada no tempo
    def normalize_and_categorize_data(self, df:pd.DataFrame, n_bins_dict='', quartis=False)->pd.DataFrame:

        normalization_type = self.NORM_TYPE

        if normalization_type == norm_type.CLUSTER:
            #listas de variáveis as quais não serão aplicadas a transformação em variação
            #incluir depois como parâmetro da função

            # donottransform = []
            # columns = df.columns
            # df[columns.drop(donottransform)] = df[columns.drop(donottransform)].diff().dropna()
            ###

            df = (df.diff()/df.shift(1)).dropna()

            output_dataframe = df.apply(self.k_means_disc, args=(n_bins_dict))

            for column in output_dataframe.columns:
                output_dataframe[column] = pd.Categorical(output_dataframe[column])

        elif normalization_type == norm_type.QUANTIS:
            output_dataframe = df.pipe(self.quantile_discretization_pipeline, quartis=quartis, n_bins_dict=n_bins_dict).dropna()

        else:
            raise Exception(f'A string {normalization_type} não é reconhecida como idetificador para um método de categorização')

        #Mudança no nome das colunas
        output_dataframe.columns = [column + ':0' for column in output_dataframe.columns]

        #Criação da coluna no tempo 1
        for column in output_dataframe.columns:
            output_dataframe.loc[:, column.split(':')[0] + ':1'] = output_dataframe[[column]].shift(-1).ffill().copy()

        return output_dataframe
    
    #Função utilizada para renomear as variáveis de uma tabela de probabilidade condicional
    #É utilizada pois as tabelas de probabilidade condicionais são geradas a partir de uma rede bayesiana clássica
    #Então os nomes são modificados para a formatação de uma rede bayesiana dinâmica
    def rename_CPD_variables(self, cpd:TabularCPD)->TabularCPD:

        new_evidence =(
        [(value.split(':')[0], int(value.split(':')[1])) for value in cpd.variables[1:]])

        new_state_names = {}
        for key in cpd.state_names.keys():
            new_state_names.update({(key.split(':')[0], int(key.split(':')[1])): cpd.state_names[key]})

        #returna uma TabularCPD com os parâmetros modificados
        return TabularCPD(variable=(cpd.variable.split(':')[0], int(cpd.variable.split(':')[1])),
                variable_card=cpd.values.shape[0],
                evidence=new_evidence,
                evidence_card=cpd.values.shape[1:],
                values=cpd.values.reshape(cpd.values.shape[0],-1),
                state_names=new_state_names)
    
    #Realiza o processo descrito nos comentários da função manterior
    #é criada uma rede classica para estimar os valores da TabularCPD
    #em seguida as tabelas são renomeadas e atribuídas a uma rede dinâmica

    def create_maximumlikelihoodestimator_based_model(self, df:pd.DataFrame,
                                                  n_bins_dict='',
                                                  quartis:bool=False)->DBN:


        topology = self.DYNAMIC_BAYESIAN_TOPOLOGY

        #O array de topologia é modificado de forma que as variáveis tenham o nome na formatação das colunas no dataframe
        
        equivalent_topology = [tuple(str(name) + ':' + str(index) for name,index in edge) for edge in topology]

        #Criação da rede clássica
        aux_bayesian_network = BayesianNetwork(equivalent_topology)


        df = df.pipe(self.normalize_and_categorize_data,
                    n_bins_dict=n_bins_dict,
                    quartis=quartis).copy()

        #criação do conjunto de nós

        node_set = set({})
        for edge in equivalent_topology:
            for node in edge:
                node_set = node_set | {node}

        #Criação da lista de CPDS
        #As CPDS são criadas e renomeadas no mesmo momento

        list_cpds_dynamic = [self.rename_CPD_variables(cpd) for cpd in
                            [MaximumLikelihoodEstimator(aux_bayesian_network, df).estimate_cpd(node) for node in node_set]]

        #Criação da rede bayesiana dinâmica
        dynamic_bayesian_network = DBN()
        dynamic_bayesian_network.add_edges_from(topology)
        dynamic_bayesian_network.add_cpds(*list_cpds_dynamic)
        dynamic_bayesian_network.initialize_initial_state()

        return dynamic_bayesian_network
    



    def create_observed_inference_file(self, savepath='modelo_arquivo_4.json'):
        n_clusters = self.N_CAT
        year_list = [int(i) for i in range(self.STARTING_YEAR + 1, self.STARTING_YEAR + self.N_YEARS + 1)]


        graph = kx.DiGraph(self.TOPOLOGY)
        exclude = [node for node in graph.nodes if list(graph.predecessors(node))]

        columns = self.BASE_DF.columns
        columns_0 = [column + ':0' for column in columns]

        inference_dict = {}
        base_normalizada = self.normalize_and_categorize_data(self.BASE_DF)

        for j, year in enumerate(year_list):
            inference_dict.update({'ANO_' + str(j):{}})
            for month in range(1,13):
                inference_dict['ANO_' + str(j)].update({'MES_' + str(month):{}})
                try:
                    estados = base_normalizada.loc[lambda x: (x.index.year==year) &(x.index.month==month)][columns_0].values[0]
                    for i, column in enumerate(columns):
                        if column not in exclude:
                            inference_dict['ANO_' + str(j)]['MES_' + str(month)].update({column:int(estados[i])})
                except IndexError:
                    inference_dict['ANO_' + str(j)].pop('MES_' + str(month))
                    continue
        
        with open(savepath, 'w') as f:
            json.dump(inference_dict, f, indent=2)



    def generate_dynamic_topology(self, time_conections=''):

        DBN_top_0 = []
        for tuple_ in self.TOPOLOGY:
            branch = []
            for element in tuple_:
                branch.append((element, 0))
            DBN_top_0.append(tuple(branch))

        DBN_top_1 = []
        for tuple_ in self.TOPOLOGY:
            branch = []
            for element in tuple_:
                branch.append((element, 1))
            DBN_top_1.append(tuple(branch))

        # node_set = set({})
        # for edge in self.TOPOLOGY:
        #         for node in edge:
        #             node_set = node_set | {node}

        # DBN_top_0_1 = [((node, 0), (node, 1)) for node in node_set]

        if time_conections=='':
            time_conections  =[((self.TARGET_VRIABLE, 0), (self.TARGET_VRIABLE, 1)), ]

        DBN_top_0_1 = time_conections

        self.DYNAMIC_BAYESIAN_TOPOLOGY = DBN_top_0 + DBN_top_0_1 + DBN_top_1

    #Retorna os resultados das predições
    #A saída é um dicionário contendo a variação numérica e a variação em categorias para cada ano
    #Na variação numérica o referrencial é o primeiro ano
    #Na variação categórica é indicada a variação com relação ao ano anterior
    #Na variação categórica também é indicada a probabilidade daquele estado
    def get_predictions(self,
                        inference_path: str,
                        n_bins_dict='',
                        quartis:bool=False,
                        ):

        df = self.BASE_DF[lambda x: x.index.year<=self.STARTING_YEAR].copy()

        with open(inference_path) as file:
            inference_dict = json.load(file)


        variable = self.TARGET_VRIABLE #variável alvo
        n_months = 12 * self.N_YEARS

        variacao_mensal_valores_numericos = {}
        variacao_mensal_categorias = {}

        parametros = self.get_norm_cat_params(df, n_bins_dict=n_bins_dict, quartis=quartis)


        model = self.create_maximumlikelihoodestimator_based_model(df, n_bins_dict=n_bins_dict, quartis=quartis)
        model.initialize_initial_state()

        #O arquivo de inferências é formatado para ser utilizado no método foward_inference
        inferences = {}
        for year in inference_dict.keys():
            for month in inference_dict[year].keys():
                for key in inference_dict[year][month].keys():
                    if (key in [node.to_tuple()[0] for node in  model.get_slice_nodes(0)]) & (key != self.TARGET_VRIABLE):
                        inferences.update({(key, 12 * int(year.split('_')[-1]) + int(month.split('_')[-1]) - 1): inference_dict[year][month][key]})

    ####### teste##########################
        #inferences = {key : inferences[key] for key in random.sample(list(inferences.keys()), k=2)}
    #######################################

        infer = DBNInference(model)

        del(model)
        #print(inferences)

        #infer.forward_inference([(variable, i) for i in range(1,n_months + 1)], inferences
        predict = infer.forward_inference([(variable, i) for i in range(1,n_months + 1)], inferences)

        del(infer)

        predictions = [predict[year].values.argmax() for year in [(variable, i) for i in range(1,n_months + 1)]]
        probability = [predict[year].values.max() for year in [(variable, i) for i in range(1,n_months + 1)]]

        upper_lim = []
        lower_lim = []
        mean = []
        #variacao_mensal_categorias#.update({'MES_' + str(mes):{}})




        ################ FOI ALTERADO AQUI
        for i in range(n_months):
            upper_lim = upper_lim + [parametros['boundaries'][variable][predictions[i]][1]]
            lower_lim = lower_lim + [parametros['boundaries'][variable][predictions[i]][0]]

            variacao_mensal_categorias.update({'MES_' + str(i): {'categoria':predictions[i], 'probabilidade': probability[i]}})

        #############################################


        ###################################################


        # val_cat_df = self.get_value_cat_df()

        # for i in range(n_months):
        #     val_min, val_max, val_mean, _ = estimador(val_cat_df, variable, categoria=predictions[i], n_projecao= 100)

        #     upper_lim = upper_lim + [val_max + (upper_lim[i-1] if i!= 0 else 0)]
        #     lower_lim = lower_lim + [val_min + (lower_lim [i-1] if i!= 0 else 0)]
        #     mean = mean + [val_mean + (mean[i-1] if i!= 0 else 0)]

        #     variacao_mensal_categorias.update({'MES_' + str(i): {'categoria':predictions[i], 'probabilidade': probability[i]}})
       
            

        #variacao_mensal_valores_numericos.update({'upper_lim': upper_lim, 'lower_lim': lower_lim, 'mean': mean})
            
        variacao_mensal_valores_numericos.update({'upper_lim': upper_lim, 'lower_lim': lower_lim})
        
        self.NUMERIC_VARI = variacao_mensal_valores_numericos

        return {'variacao_numerica': variacao_mensal_valores_numericos,
                'variacao_categoria': variacao_mensal_categorias}
    


    #Retorna o limite superior e inferior da previsão (agora em valor esperado em vez de variação esperada)
    #Mostra um plot das previsões se plot=True
    def show_predictions(self, plot=False, savepath='default.svg'):

        df = self.BASE_DF

        n_years = self.N_YEARS

        prediction_dict = self.NUMERIC_VARI

        valor_atual = df[lambda x: x.index.year<=2017][self.TARGET_VRIABLE].iloc[-1]

        upper_lim = prediction_dict['upper_lim']
        lower_lim = prediction_dict['lower_lim']
        #mean = prediction_dict['mean']


        x_ = df[lambda x: x.index.year<=2017].index
        last_month = x_[-1]
        #last_month = df.index[-1]

        # upper = [valor_atual] + list(np.array(upper_lim) + valor_atual)
        # lower = [valor_atual] + list(np.array(lower_lim) + valor_atual)
        #mean = [valor_atual] + list(np.array(mean) + valor_atual)


        ###########
        upper = [valor_atual]
        lower = [valor_atual]

        for percentage in np.array(upper_lim):
            upper.append(upper[-1] + upper[-1]*percentage)

        for percentage in np.array(lower_lim):
            lower.append(lower[-1] + lower[-1]*percentage)

        #######

        df_previsoes = pd.DataFrame(columns=['upper_lim', 'lower_lim'])

        df_previsoes['upper_lim'] = pd.Series(upper, index=pd.date_range(last_month,periods=12*self.N_YEARS+1,freq='MS')) #alterar pois depende do nímero de meses previstos
        df_previsoes['lower_lim'] = pd.Series(lower, index=pd.date_range(last_month,periods=12*self.N_YEARS+1,freq='MS')) #alterar pois depende do número de meses previstos
        #df_previsoes['mean'] = pd.Series(mean, index=pd.date_range(last_month,periods=12*self.N_YEARS+1,freq='MS'))


        #x = [last_month + i*datetime.timedelta(days=30) for i in range(len(upper))]

        df_previsoes['mean'] = (df_previsoes['upper_lim'] + df_previsoes['lower_lim'])/2

        if plot:
            plt.plot(df_previsoes['upper_lim'].index, df_previsoes['upper_lim'].values, label='Limite Superior', linestyle='--', alpha=0.7, linewidth=0.7)
            plt.plot(df_previsoes['lower_lim'].index, df_previsoes['lower_lim'].values, label='Limite Inferior', linestyle='--', alpha=0.7, linewidth=0.7)
            plt.plot(df_previsoes['mean'].index, df_previsoes['mean'].values, label='Média')
            plt.plot(df[self.TARGET_VRIABLE].index, df[self.TARGET_VRIABLE].values, label='Valor Observado')
            #plt.plot(x[:2], df2.loc[lambda x: x.index.month == month_index]['PLD'].values)
            #plt.ylim((0, 1000))
            #_ = plt.xticks((x_ + x[1:])[::2])
            plt.legend(['Limite Superior','Limite Inferior', 'Média', 'Valor Observado'])
            #plt.xlabel('Ano')
            #plt.ylabel('Consumo Faturado (MWh)')
            #plt.savefig(savepath)
            plt.show()


        prev_ob = pd.merge(df_previsoes[['mean','upper_lim','lower_lim']].resample('MS').mean(), self.BASE_DF[[self.TARGET_VRIABLE]].resample('MS').mean(), how='left', left_index=True, right_index=True).dropna()

        return prev_ob.dropna()
    
    def get_cat_limits(self):
        limits_dict = self.get_norm_cat_params(self.BASE_DF)
        f_limits_dict = {}

        for variable in limits_dict['base_values'].index:
            f_limits_dict.update({variable: {}})
            for state in range(self.N_CAT):
                f_limits_dict[variable].update({state:[value*limits_dict['base_values'][variable] for value in limits_dict['boundaries'][variable][state]]})
        
        return f_limits_dict
    
    def get_value_cat_df(self):
        df = self.BASE_DF
        categories_df = self.normalize_and_categorize_data(df)

        out_df = (df.diff()/df.shift(1)).dropna()
        
        for column in df.columns:
            out_df.insert(loc=out_df.columns.get_loc(column) + 1, column=column + '_CAT', value=categories_df[column + ':0'])

        return out_df.dropna()
