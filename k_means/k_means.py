import random
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k=5):
        self.k = k

    def load_dataset(self, path):
        dataset = []
        f = open(path)
        for line in f.readlines():
            d = line.strip().split(',')
            dataset.append(d)

        random.shuffle(dataset)

        for i in range(len(dataset)):
            dataset[i] = (list(map(float, dataset[i][:-1])))

        print('dataset:', dataset)
        self.dataset = dataset
        return dataset

    def _calculate_dist(self, a, b):
        return np.sqrt(np.sum(np.power(a - b, 2)))

    def create_centroids(self, dataset):
        # num of features
        n = np.shape(dataset)[1]
        dataset = np.array(dataset)
        centroids = np.mat(np.zeros((self.k, n)))

        for j in range(n):
            min_j = min(dataset[:, j])
            max_j = max(dataset[:, j])
            range_j = float(max_j - min_j)
            centroids[:, j] = min_j + range_j * \
                              np.random.rand(self.k, 1)  # here k*1 vector

        self.centroids = centroids
        return centroids

    def train(self, dataset, epochs):
        dataset = np.array(dataset)
        centroids = self.create_centroids(dataset)
        m = np.shape(dataset)[0]  # dataset's count
        cluster_data = np.mat([[0, 0] for i in range(m)])

        # every train epoch
        for e in range(epochs):
            print('Epoch:', e + 1)
            # for every data in dataset
            for i in range(m):
                cent_dist = []  # centroid index-distance list
                # for every centroid,calculate each centroid's distance to data
                for j in range(self.k):
                    dist = self._calculate_dist(centroids[j, :], dataset[i, :])
                    cent_dist.append((j, dist))
                # choose minimum distance's centroid as data's cluster centroid
                min_index, min_dist = min(cent_dist, key=lambda x: x[1])

                if cluster_data[i, 0] != min_index:
                    print('cluster changed!')

                cluster_data[i] = [min_index, min_dist]

            # update centroids' position
            for j in range(self.k):
                # np.zeros(...) bool type = true,return index
                pts = dataset[np.nonzero(cluster_data[:, 0].A == j)[0]]
                centroids[j, :] = np.mean(pts, axis=0)

        return centroids, cluster_data

    def draw_kmeans_cluster(self, dataset, cendroids, cluster_data):
        dataset = np.array(dataset)
        fig = plt.figure('K-Means')
        color = ['red', 'blue', 'green', 'yellow', 'black']

        for j in range(self.k):
            data = dataset[np.nonzero(cluster_data[:, 0].A == j)[
                0]].tolist()  # convert to list
            # maybe data's class<k,so data could be 0
            if data:
                plt.scatter(data[0], data[1], s=30, label='class' +
                                                          str(j), c=color[j])
            plt.scatter(cendroids[j, 0], cendroids[j, 1],
                        s=30, c=color[j], marker='+')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.title('K-Means')
        plt.show()


def main():
    kmeans = KMeans(k=3)
    dataset = kmeans.load_dataset('F:\project\python\ML\k_means\iris.data')
    cendroids, cluster_data = kmeans.train(dataset, epochs=1)
    kmeans.draw_kmeans_cluster(dataset, cendroids, cluster_data)


if __name__ == "__main__":
    main()
