# Abaixo, colocarei todos os códigos
# Para instalação das bibliotecas que utilizei um ambiente virtual para instalar as bibliotecas no
# Ubuntu, para poder rodar o código com o suporte nativo

# Obs: Fui comentando conforme fui fazendo o código, para facilitar o meu aprendizado fazendo, e para caso precise
# ler o passo a passo.

# ----------------------------- Pip Installs -----------------------------
# pip install kagglehub
# pip install pandas
# pip install numpy
# pip install scikit-learn
# ----------------------------- Pip Installs -----------------------------

import pandas as pd
import numpy as np

# Ao invés de baixar o arquivo manualmente, utilizei a biblioteca kagglehub para baixar o dataset direto do Kaggle
import kagglehub
path = kagglehub.dataset_download("ibrahimqasimi/world-bank-population-total")
print("Path to dataset files:", path)

df = pd.read_csv(f"{path}/wb_population_total.csv")

# Aqui ajustei o dataframe para ficar mais fácil de trabalhar com ele
# Ajustei os nomes das colunas, converti os valores para numéricos, e removi valores nulos.
df = df.rename(columns={
    "countryiso3code": "iso3",
    "date": "year",
    "value": "population",
    "country.value": "country"
})
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["population"] = pd.to_numeric(df["population"], errors="coerce")
df = df.dropna(subset=["population", "year"])


# Aqui, eu criei três classes para fazer a caracterização da população dos países
# Pequena tem até 10 milhões de habitantes, média é entre 10 e 100 milhões, e grande é acima de 100 milhões.
bins = [0, 10_000_000, 100_000_000, np.inf]
labels = ["Pequena", "Média", "Grande"]
df["class_name"] = pd.cut(df["population"], bins=bins, labels=labels, right=False)
class_map = {"Pequena": 0, "Média": 1, "Grande": 2}
df["class_id"] = df["class_name"].map(class_map)

# Abaixo, foram criadas outras features e atributos, elas focam em três pontos principais.
# Uma, mantendo o year como atributo, e outros dois focando na taxa de crescimento, e na população do ano anterior.
df = df.sort_values(["iso3", "year"])
df["population_lag_1"] = df.groupby("iso3")["population"].shift(1)
df["growth_rate"] = (df["population"] - df["population_lag_1"]) / df["population_lag_1"]
df_model = df.dropna(subset=["population_lag_1", "growth_rate", "class_id"]).copy()

feature_cols = ["year", "population_lag_1", "growth_rate"]
X = df_model[feature_cols]
y = df_model["class_id"].astype(int)

# A partir daqui realizei o treino da árvore de decisão, com as bibliotecas já instaladas anteriormente.
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# X são os atributos, e Y os rótulos, tendo as variáveis, e as labels, que são as classes que queremos prever.
# 20% dos dados vão para teste com test_size 0.2, e 80% para treino.
# Random_state é a semente para ser aleatório, garantindo que sejam reproduziveís novamente, dividindo por 42
# Stratify vai garantir que as classes no treino e no teste sejam mantidas
# Logo, o train vai treinar o modelo, e o test vai ser usado para testar o modelo em dados que não foram vistos ainda.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# o clf é usado para prever as classes dos exemplos de teste, e o resultado y_pred 
# É um array com as classes que foram previstas de forma numérica. Com o 0, 1 e 2.
# Também calculei a acurácia da previsão, mostrando no relatório:
# Precisão, sendo a porcentagem de acertos, 
# e (Revocação/Sensibilidade), o qual quer dizer de quantos que eram X o modelo achou, por exemplo
# F1-Score, que é uma média entre precisão e revocação.
# E por fim, o supporte, que são números de exemplos reais do teste.
# A matriz de confusão, vai mostrar por fim, onde o modelo acertou, e onde ele errou por classe. Com a confusão
# Quando ele errou, e para qual classe ele errou.
y_pred = clf.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(
    y_test, y_pred, target_names=labels, zero_division=0))
print("\nMatriz de confusão:\n", confusion_matrix(y_test, y_pred))

# Aqui eu tento, com novos valores, e um dataframe criado, fazer uma previsão de como serão
# Os valores futuros.
novo_exemplo = pd.DataFrame({
    "year": [2025],
    "population_lag_1": [12_000_000], 
    "growth_rate": [0.015] 
})
pred = clf.predict(novo_exemplo)[0]
inv_map = {v:k for k,v in class_map.items()}
print("\nPrevisão para o exemplo:", inv_map[int(pred)])

# Aqui, o modelo aprendeu a prever qual a classe de população que ele irá ser, com base nos dados que ele possui,
# Utilizando a população do ano anterior, crescimento da população, e o ano.