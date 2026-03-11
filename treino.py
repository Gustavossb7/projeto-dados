import pandas as ps
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

base = ps.read_csv('titanic_train.csv')

## preenchendo valores vazios com a mediana da idade

# Extrai o título que vem antes do ponto final no nome
base['Title'] = base['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

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

print(base.info)
print(base)
#criei uma nova coluna 'Deck' e utilizei uma regex para selecionar só o caracter não numérico da coluna 'Cabin'
base['Deck']  = base['Cabin'].replace('[^A-Z]','',regex=True).astype('string')

base['Deck'] = base['Deck'].fillna('U')
print(base)

