import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def encode_class(df,lista_class):
    x = {}
    for i in range(len(lista_class)):           #criando dicionário que ajudará a simplificar o código do próximo loop
        x[lista_class[i]] = i
    for i in range(len(df.index)):
        df.at[i,'class'] = x[df.at[i,'class']]
   

    
    


dados = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header = None)
print(dados)
dados.columns =  ['sepal_length','sepal_width','petal_length','petal_width','class']
print("Colunas:")
for c in dados.columns:
    print(c)
print("Numero de linhas: " + str(len(dados.index)))
print('')       #Quebra de linha
print("Tipos das colunas:")
print(dados.dtypes)
print('')
print("VALORES MAXIMOS:")
print(dados[['sepal_length','sepal_width','petal_length','petal_width']].max())
print('')
print("VALORES MINIMOS:")
print(dados[['sepal_length','sepal_width','petal_length','petal_width']].min())
print('')
print("VALORES MEDIOS:")
print(dados[['sepal_length','sepal_width','petal_length','petal_width']].mean())
print('')
print("DESVIOS PADRAO:")
print(dados[['sepal_length','sepal_width','petal_length','petal_width']].std())
print('')
lista_class = dados['class'].unique()       #Lista com valores unicos de class
print('LISTA COM VALORES UNICOS DE CLASS:')
print(lista_class)
print('')
print('DADDOS ANTES DE CLASSES SEREM CODIFICADOS:')
print(dados)
print('')
print('DADOS DEPOIS DE CLASSES SEREM CODIFICADOS:')
dados_com_numero = encode_class(dados,lista_class)
print(dados)
print('')
dados_auxiliar = pd.DataFrame(columns = ['sepal_length','sepal_width','petal_length','petal_width','class']) #Dataframe auxiliar que terá apenas os dados com classe 1 e 2
dados_auxiliar = dados.loc[(dados['class'] == 1) | (dados['class'] == 2)]
print('DATAFRAME AUXILIAR:')
print(dados_auxiliar)
print('')
dados_auxiliar.plot.scatter(x = 'sepal_length', y = 'petal_length') 
plt.show()                                                          

dados_auxiliar.plot.scatter(x = 'sepal_length', y = 'petal_length',c ='class',colormap = 'jet') 
plt.show()                                                                                      
                                                                                                
