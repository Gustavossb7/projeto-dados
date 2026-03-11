import pandas as ps
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders.one_hot import OneHotEncoder


base = ps.read_csv('titanic_train.csv')

## preenchendo valores vazios com a mediana da idade

# Extrai o título que vem antes do ponto final no nome
base['Title'] = base['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Agrupa por título e calcula a mediana da idade
medians_by_title = base.groupby('Title')['Age'].median()

# Preenche os NaNs da idade com a mediana do seu respectivo título
base['Age'] = base['Age'].fillna(base.groupby('Title')['Age'].transform('median'))


## Identificando outliers

# Calculando os quartis
Q1 = base['Fare'].quantile(0.25)
Q3 = base['Fare'].quantile(0.75)
IQR = Q3 - Q1

# Definindo os limites
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Identificando quem são os outliers
outliers = base[base['Fare'] > limite_superior]
print(f"Existem {len(outliers)} passageiros que pagaram valores muito acima da média.")


## separando coluna cabin


#criei uma nova coluna 'Deck' e utilizei uma regex para selecionar só o caracter não numérico da coluna 'Cabin'
base['Deck']  = base['Cabin'].replace('[^A-Z]','',regex=True).astype('string')

#Todos os NaN são substituidos por 'U' de Unknown
base['Deck'] = base['Deck'].fillna('U')



## aplicando label encoder e one hot enconder

#aplicando label encoder na coluna 'Sex'
label_encoder = LabelEncoder()
base['Sex'] = label_encoder.fit_transform(base['Sex'])

# Criando colunas extras para Embarked e Deck (One-Hot Encoding)
base = ps.get_dummies(base, columns=['Embarked', 'Deck'])

# Removendo o que sobrou de "lixo" (colunas que não são números)
base.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True, errors='ignore')


## escalonamento

scaler = StandardScaler()

base[['Age', 'Fare']] = scaler.fit_transform(base[['Age', 'Fare']])

print(base)