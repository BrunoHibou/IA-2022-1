from turtle import distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pandas
from sklearn.cluster import KMeans
from sympy import Sum
from yellowbrick.cluster import KElbowVisualizer

np.random.seed(3)


def calculate_euclid(A, B):
    # dados dois numpy arrays calcula a distancia euclideana
    euclidean = np.square(np.sum((A - B) ** 2))
    return euclidean


def wcss(A, B):
    # dados dois numpy arrays calcula a distancia euclideana
    wcss = np.sum((A - B) ** 2)
    return wcss


class kMeans:
    def __init__(self, k=5, max_iters=300, plot_steps=False):
        self.k = k  # quantidade de clusters
        self.max_iters = max_iters  # quantidade máxima e iterações
        self.plot_steps = plot_steps  # vai plotar cada passo?

        # lista de índices para cada cluster
        self.clusters = [[] for _ in range(self.k)]

        # os centroides de cada cluster
        self.centroids = []

    def predict(self, x):
        self.x = x
        self.n_samples, self.n_features = x.shape

        # inicializaçãosr
        random_indexes = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.x[index] for index in random_indexes]

        # otimização dos clusters
        for _ in range(self.max_iters):
            # atualiza os clusters
            self.clusters = self._create_clusters(self.centroids)

            # atualiza os centróides
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # verifica se os clusters sofreram alteração
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()
        self.plot()

        # Cclassifica cada amostra com o index do cluster
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # Cada amostra é associada a um centróide
        labels = np.empty(self.n_samples)

        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_index
        return labels

    def _create_clusters(self, centroids):
        # atribui cada amostra a um centróide criando os clusters
        clusters = [[] for _ in range(self.k)]
        for index, sample in enumerate(self.x):
            centroid_index = self._closest_centroid(sample, centroids)
            clusters[centroid_index].append(index)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # calcula a distância entre o ponto atual e todos os centróides e retorna o index do mais próximo
        distances = [calculate_euclid(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # atribui o valor do "centro de gravidade" dos clusters aos centróides
        centroids = np.zeros((self.k, self.n_features))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.x[cluster], axis=0)
            centroids[cluster_index] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # compara as distâncias dos centróides da última iteração com a atual
        distances = [
            calculate_euclid(centroids_old[i], centroids[i]) for i in range(self.k)
        ]
        return sum(distances) == 0

    # plota o gráfico baseado nos centróides atuais
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.x[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()
        print(self.centroids)


class Elbow:
    def __init__(self) -> None:
        self.km = np.ndarray.tolist(np.zeros((10)))
        self.y_pred = []

        # aqui estamos lendo a base de dados
        self.df = pandas.read_csv("wine.csv").dropna()

        # plotando a base de dados em um gráfico
        self.array = np.column_stack([self.df["Hue"], self.df["Alkalinity."]])

        for p in range(10):
            self.km[p] = kMeans(k=p + 1, max_iters=150, plot_steps=False)
            self.y_pred = self.km[p].predict(self.array)

    def elbow_distances(self, sample, centroids):
        distances = [calculate_euclid(sample, point) for point in centroids]
        sum = 0
        for p in distances:
            sum = p**2 + sum
        return sum

    def elbowM(self):
        elbow = [[]]

        for kmeans in self.km:
            sum = self.elbow_distances(kmeans.x, kmeans.centroids)
            elbow[0].append(sum)
        elbow.append(range(10))
        plt.plot(elbow[1], elbow[0], "bx-")
        plt.show()


# Testes
if __name__ == "__main__":

    df = pandas.read_csv("iris.csv").dropna()
    array = np.column_stack([df["sepal length in cm"], df["petal length in cm"]])

    # print(array)

    k = kMeans(k=3, max_iters=150, plot_steps=False)
    y_pred = k.predict(array)

    # elbow = Elbow()
    # print("a")
    # elbow.elbowM()
