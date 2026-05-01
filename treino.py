import pandas as ps
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders.one_hot import OneHotEncoder
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
#print(f"Existem {len(outliers)} passageiros que pagaram valores muito acima da média.")


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
base = ps.get_dummies(base, columns=['Embarked', 'Deck', 'Title'])

# Removendo o que sobrou de "lixo" (colunas que não são números)
base.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True, errors='ignore')


## escalonamento

scaler = StandardScaler()
base[['Age', 'Fare']] = scaler.fit_transform(base[['Age', 'Fare']])

print(base)

##Train-Test Split

#'Survived' sera a coluna alvo
y = base['Survived']

# As demais colunas serão as features
X = base.drop('Survived', axis=1)

#80% da base de dados como treino e 20% como teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Conferindo o tamanho das fatias
print(f"Total de passageiros: {len(X)}")
print(f"Passageiros para treino: {len(X_train)}")
print(f"Passageiros para teste: {len(X_test)}")

##Árvore de decisão

#criando classificador, utilizei max_depth=3 para evitar Overfitting
clf = tree.DecisionTreeClassifier(max_depth=3, random_state=42)

#fazendo o fit com os dados de treino (Treinando o modelo, ele está estudando)
clf.fit(X_train, y_train)

#usando o predict (Modelo está criando suas respostas)
y_pred = clf.predict(X_test)

#comprarando o y_test com o y_pred (Comparando as respostas do modelo com o gabarito)
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia da Árvore: {acuracia * 100:.2f}%")

#Gráfico da árvore de decisão
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['Morreu', 'Sobreviveu'], filled=True)
plt.show()

#Criação da matrix de confusão
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Morreu', 'Sobreviveu']).plot(cmap='Blues')


#Análise de relevância das colunas
feat_importances = ps.Series(clf.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh', title='As 5 Variáveis Mais Importantes')
plt.show()


##Random forest

clf = RandomForestClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print(f"Random forest accuracia: {acuracia * 100:.2f}%")


##Regressão logistica

# Criando Pipeline 
pipeline_log = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', LogisticRegression())
])

# Validando com Cross-Validation (5 dobras)
scores = cross_val_score(pipeline_log, X, y, cv=5)
print(f"Acurácia Média (Cross-Val): {scores.mean() * 100:.2f}%")

# Treinando o pipeline para acessar o modelo interno
pipeline_log.fit(X_train, y_train)

# Acessando os coeficientes dentro do pipeline
modelo_final = pipeline_log.named_steps['model']
coeficientes = ps.DataFrame(modelo_final.coef_[0], index=X.columns, columns=['Peso'])
coeficientes = coeficientes.sort_values(by='Peso', ascending=False)


y_pred_log = pipeline_log.predict(X_test)
acuracia = accuracy_score(y_test, y_pred_log)
print(f"Regressão logistica accuracia: {acuracia * 100:.2f}%")