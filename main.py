import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from elbow import elbow

# Carregando o conjunto de dados 'mpg' do Seaborn
dados = sns.load_dataset('mpg')
df = pd.DataFrame(dados)

# Selecionando um subconjunto aleatório de 20 linhas
df_subconjunto = df.sample(30, random_state=42)

# Seleção das colunas relevantes
colunas_numericas = ['mpg', 'horsepower', 'weight']
dados_para_aglomeracao = df_subconjunto[colunas_numericas]

# Imputação dos valores NaN usando a média
dados_para_aglomeracao.fillna(dados_para_aglomeracao.mean(), inplace=True)

# Chamar a função elbow com as linhas selecionadas (sem NaN)
elbow(dados_para_aglomeracao)

# Remover linhas NaN antes de aplicar K-Means e dendrograma
dados_para_aglomeracao_sem_nan = dados_para_aglomeracao.dropna()

# Número desejado de clusters
num_clusters = 4
limite_distancia = 1000

# Aplicação do algoritmo K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df_subconjunto['cluster'] = kmeans.fit_predict(dados_para_aglomeracao_sem_nan)

# Cálculo da matriz de ligação usando a distância euclidiana
matriz_ligacao = linkage(dados_para_aglomeracao_sem_nan, method='ward')

# Criação e exibição do dendrograma com cores dos clusters
plt.figure(figsize=(15, 5))
dendrogram(matriz_ligacao, labels=df_subconjunto['name'].tolist(), leaf_rotation=90, leaf_font_size=8, color_threshold=limite_distancia)
plt.title('Dendrograma de Aglomeração Hierárquica com Clusters K-Means')
plt.xlabel('Marca')
plt.ylabel('Rendimento')
plt.show()

# Exibir os clusters resultantes
print("Clusters:")
print(df_subconjunto[['name', 'cluster']])
