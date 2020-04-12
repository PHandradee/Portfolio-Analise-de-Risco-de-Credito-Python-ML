# O objetivo deste modelo é analisar o perfil dos clientes para verificar se é
#ou não um bom pagador, a taxa de acerto mínima aceita é de 75%, usando NaiveBayes

#carregando dados
import pandas as pd

dados = pd.read_csv("Credito.csv", sep = ";", encoding = "cp860") 
#sep é o separador dos dados";", encoding é porque os dados estão em pt

#-----------------------------------------------------------------------------

#Separando Variáveis
X = dados.iloc[:,0:19].values #iloc é para selecionar dados, aqui selecionando
# todas as linhas":", e da coluna 0 até a 19.É colocado .values para resgatar
#apenas os valores deste intervalo, sem o nome das colunas. Variáveis Independentes

Y = dados.iloc[:,19].values # mesma questão acima, porém aqui é selecionado apenas 
# a classe

#-----------------------------------------------------------------------------

# Transformando atributos categóricos(string) em atributos numéricos
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:,0]= labelencoder.fit_transform(X[:,0])
X[:,2]= labelencoder.fit_transform(X[:,2])
X[:,3]= labelencoder.fit_transform(X[:,3])
X[:,5]= labelencoder.fit_transform(X[:,5])
X[:,6]= labelencoder.fit_transform(X[:,6])
X[:,8]= labelencoder.fit_transform(X[:,8])
X[:,9]= labelencoder.fit_transform(X[:,9])
X[:,11]= labelencoder.fit_transform(X[:,11])
X[:,13]= labelencoder.fit_transform(X[:,13])
X[:,14]= labelencoder.fit_transform(X[:,14])
X[:,16]= labelencoder.fit_transform(X[:,16])
X[:,18]= labelencoder.fit_transform(X[:,18])
# Foram selecionadas as colunas onde existiam valores categóricos e usados Label Encoder
#para converter para númericos

#-----------------------------------------------------------------------------

#Dividindo a base de dados em teste e treinamento.
from sklearn.model_selection import train_test_split

X_treinamento,X_teste,Y_treinamento,Y_teste = train_test_split(X,Y,test_size=0.3,
                                                               random_state=0)
#test_size é o tamanho das amostras, sendo 70% e outra 30%, random_state é para
#não ficar gerando valores aleatórios de amostra

#-----------------------------------------------------------------------------

#Selecionando os atribudos mais relevantes
from sklearn.ensemble import ExtraTreesClassifier

selec_atributos = ExtraTreesClassifier()
selec_atributos.fit(X_treinamento,Y_treinamento)
atributos = selec_atributos.feature_importances_
#.fit é para treinar o modelo. feature_importances_ função para detectar a importancia dos atributos
X_teste2 = X_teste[:,[0,1,3,4,5,6,12]]
X_treinamento2 = X_treinamento[:,[0,1,3,4,5,6,12]]

#-----------------------------------------------------------------------------

#Usando outro algoritmo de arvore para resolver o modelo

from sklearn.svm import SVC

svm = SVC()
svm.fit(X_treinamento2,Y_treinamento)
previsao = svm.predict(X_teste2)

#-----------------------------------------------------------------------------

#Verificando a taxa de acerto

from sklearn.metrics import accuracy_score

taxa_acerto = accuracy_score(previsao,Y_teste)

#Tivemos taxa de acerto de 71,66%, sendo ainda inferior aos 75% necessários.
