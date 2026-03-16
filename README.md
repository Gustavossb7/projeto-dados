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

### 6. Divisão dos Dados (Train-Test Split)
Para garantir uma avaliação imparcial do modelo, utilizei a técnica de Train-Test Split. Os dados foram divididos na proporção de 80% para treinamento e 20% para teste, utilizando o random_state=42 para garantir a reprodutibilidade dos resultados.

Variável Alvo (y): Survived (indica se o passageiro sobreviveu ou não).

Variáveis Preditoras (X): Todas as demais características processadas (Idade, Sexo, Classe, etc.).

### 7. Árvore de Decisão
O primeiro algoritmo escolhido foi a Árvore de Decisão. Para evitar o Overfitting (quando o modelo decora os dados de treino mas não generaliza para novos dados), limitei a profundidade máxima da árvore (max_depth) em 3.

Após realizar o fit com os dados de treino e o predict com os dados de teste, o modelo alcançou uma Acurácia de 81%.

#imagem da árvore

### 8. Matriz de Confusão
Para ir além da acurácia, gerei uma Matriz de Confusão. Ela permite visualizar o desempenho do algoritmo em cada categoria, identificando:

Verdadeiros Positivos/Negativos: Acertos do modelo.

Falsos Positivos/Negativos: Erros de classificação (onde o modelo se "confundiu").

#imagem matriz

### 9. Análise de Relevância (Feature Importance)
Utilizei o atributo feature_importances_ para extrair quais variáveis foram mais determinantes para as decisões do modelo. Esta etapa é fundamental para a Explicabilidade da IA.

#imagem da análise de relevância

Como demonstrado no gráfico, as variáveis de maior peso foram o Título (que sintetiza gênero e idade) e a Classe social.

##🏆 Conclusão
Neste projeto, percorri todo o pipeline de Ciência de Dados: desde a extração e limpeza (Data Cleaning) até a engenharia de recursos (Feature Engineering) e modelagem.

O modelo de Árvore de Decisão confirmou, através dos dados, a máxima histórica do desastre: passageiros com títulos femininos/infantis e aqueles hospedados em classes superiores tiveram chances significativamente maiores de sobrevivência. O projeto resultou em um modelo robusto com 81% de precisão em dados nunca vistos.
