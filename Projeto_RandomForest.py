import pandas as pd

dados = pd.read_csv("Credito.csv", sep=";", encoding = "cp860")

previsores = dados.iloc[:,0:19].values
classe = dados.iloc[:,19].values

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder.fit_transform(previsores[:,6])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder.fit_transform(previsores[:,9])
previsores[:,11] = labelencoder.fit_transform(previsores[:,11])
previsores[:,13] = labelencoder.fit_transform(previsores[:,13])
previsores[:,14] = labelencoder.fit_transform(previsores[:,14])
previsores[:,16] = labelencoder.fit_transform(previsores[:,16])
previsores[:,18] = labelencoder.fit_transform(previsores[:,18])

from sklearn.model_selection import train_test_split

X_treinamento,X_teste,Y_treinamento,Y_teste = train_test_split(previsores,classe,test_size=0.3,
                                                               random_state=0)

from sklearn.ensemble import ExtraTreesClassifier

arvore = ExtraTreesClassifier()
arvore.fit(X_treinamento,Y_treinamento)
atributos = arvore.feature_importances_


X_treinamento2 = X_treinamento[:,[0,1,2,3,4,6,12]]
X_teste2 = X_teste[:,[0,1,2,3,4,6,12]]

from sklearn.ensemble import RandomForestClassifier

modelo = RandomForestClassifier(n_estimators=130,min_samples_split=2,random_state=1)
modelo.fit(X_treinamento,Y_treinamento)
previsao = modelo.predict(X_teste)

from sklearn.metrics import accuracy_score

taxa_acerto = accuracy_score(Y_teste,previsao)