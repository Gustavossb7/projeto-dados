# 🚢 Titanic Data Science Project: Preparação de Dados
Este repositório contém a etapa de Data Cleaning e Feature Engineering do dataset do Titanic, seguindo as premissas do CRISP-DM para garantir que os dados estejam prontos para modelos de Machine Learning.

## 🛠️ Etapas do Projeto
### 1. Tratamento de Dados Faltantes (Age)
Inicialmente, a estratégia comum seria aplicar a mediana geral (28 anos) em todos os valores nulos. No entanto, ao analisar a coluna Name, notei a presença de títulos (Master, Miss, Mr, etc.).

Raciocínio: Aplicar uma média única poderia "envelhecer" crianças (Master) ou "rejuvenescer" idosos. Optei por extrair o título e aplicar a mediana específica de cada grupo.

Resultado: A mediana para Master foi de 3.5 anos, enquanto para Mr foi de 30.0, tornando os dados muito mais assertivos.

### 2. Análise de Outliers (Fare)
Utilizei o método do Intervalo Interquartil (IQR) para identificar valores discrepantes na coluna de tarifas.

Estratégia: Embora identificados, optei por não remover os outliers. Em um cenário de naufrágio, o poder aquisitivo (tarifas mais altas) é uma variável crítica para a probabilidade de sobrevivência ("Mulheres e crianças da primeira classe primeiro").

### 3. Feature Engineering: Criando a coluna Deck
A coluna Cabin continha informações complexas. Extraí apenas a primeira letra para identificar o Deck (andar do navio) e tratei valores nulos como 'U' (Unknown).

### 4. Codificação de Variáveis Categóricas
Para que os algoritmos matemáticos processem os textos, utilizei duas técnicas distintas:

Label Encoding: Aplicado na coluna Sex (transformando em 0 e 1).

One-Hot Encoding (get_dummies): Aplicado em Embarked e Deck, criando colunas binárias para evitar que o modelo interprete uma ordem de importância inexistente entre os portos ou andares.

### 5. Escalonamento de Variáveis (Scaling)
Para evitar que o modelo seja enviesado por colunas com grandezas diferentes (como Fare, que chega a 512, e Age, que chega a 80), utilizei o StandardScaler.

Objetivo: Colocar todos os dados na mesma escala (média 0 e desvio padrão 1), garantindo que a IA dê a importância correta a cada característica.

## 🚀 Próximos Passos
Com os dados limpos, codificados e escalonados, o projeto segue para a Fase 4: Modelagem, onde testarei algoritmos de classificação como Árvores de Decisão e Regressão Logística.
