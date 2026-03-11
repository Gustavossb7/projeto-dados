import pandas as ps
import seaborn as sns
import matplotlib.pyplot as plt


base = ps.read_csv('titanic_train.csv')
base.head() # primeiras 5 linhas
base.tail() # ultimas 5 linhas
base.shape #linhas por colunas

base.Survived #Seleciona uma coluna
base['Survived']

base.Survived.value_counts() #conta a quantidade de cada valor da coluna

base.describe #seleciona alguns dados importantes

base[base.Survived == 1]

dados = { #criando um dicionário de arrays
    'x': [1,2,3], 'y':['pizza', 'hamburguer', 'batata']
}

dado = [1,2,3]

dados = ps.DataFrame(dados) #transforma dados em dataframes

#print(dados.mean()) #média



#print(base[base.Survived == 1] & base[base.Pclass == 1])



sns.pairplot(base, hue='Survived') #cria alguns gráficos
#plt.show()

base = base.drop(['PassengerId','Name'],axis=1) #eliminando as colunas name e passengerId
#base = base.dropna() #eliminando todas as linhas com algum valor vazio

#print(base.info()) #informações dos dados
#print(base.describe()) #informações estatisticas

(base['Survived'].value_counts())#quantos vlaores de cada tem

print(base.sort_values(by='Fare', ascending=False))

print(base)
idade_media = base['Age'].median()
base['Age'] = base['Age'].fillna(base['Age'].median())
print(idade_media)