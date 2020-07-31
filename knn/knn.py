import math
import random
import numpy as np
import pandas as pd


class KNN:
    def __init__(self, k=20):
        self.k = k
        self.dataset = None
        self.label = None

    def load_dataset(self, path):
        # load dataset from file
        dataset, label = [], []
        f = open(path)
        for line in f.readlines():
            d = line.strip().split(',')
            dataset.append(d)

        random.shuffle(dataset)
        for i in range(len(dataset)):
            label.append(dataset[i][-1])
            dataset[i] = (list(map(float, dataset[i][:-1])))

        '''
        data=pd.read_table(data_path,sep=',',header=None)
        # convert pandas dataframe to python list
        data=np.array(data).tolist()
        # shuffle the data list
        random.shuffle(data)
        '''

        print('dataset:', dataset)
        print('label:', label)
        self.dataset, self.label = dataset, label
        return dataset, label

    def _get_k_nearest_neighbors(self, train_data, train_label, test):
        # get top k nearest neighbors' list
        neighbors = []
        for i in range(len(train_data)):
            # calculate distance
            sq_diff = 0
            for n in range(len(train_data[i])):
                sq_diff += (train_data[i][n] - test[n]) ** 2
            distance = math.sqrt(sq_diff)

            neighbors.append((distance, train_label[i]))

        neighbors.sort(key=lambda x: x[0])  # sort neighbors list by distance

        # select top 'k' nearest neighbors
        k_nearest_neighbors = neighbors[:self.k]
        return k_nearest_neighbors

    def _get_highest_proportion_neighbor(self, k_nearest_neighbors):
        # sort k nearest neighbors and choose highest proportion neighbor as prediction
        dic = {}
        for neighbor in k_nearest_neighbors:
            if neighbor[1] in dic:
                dic[neighbor[1]] += 1
            else:
                dic[neighbor[1]] = 1
        # sort by num
        lst = sorted(dic.items(), key=lambda x: x[1], reverse=True)

        return lst[0][0]

    def predict(self, train_data, train_label, test_data, test_label):
        pre_lable = []

        for test in test_data:
            k_nearest_neighbors = self._get_k_nearest_neighbors(
                train_data, train_label, test)
            pre = self._get_highest_proportion_neighbor(k_nearest_neighbors)
            pre_lable.append(pre)
        return pre_lable

    def evaluate(self, train_data, train_label, test_data, test_label):
        pre_label = self.predict(
            train_data, train_label, test_data, test_label)
        wrong = 0
        for i in range(len(test_label)):
            if test_label[i] != pre_label[i]:
                wrong += 1
        accuracy = (1 - wrong / len(test_data)) * 100
        print('Accuracy is %.1f%%' % accuracy)


def main():
    knn = KNN(k=20)
    dataset, label = knn.load_dataset('F:\project\python\ML\knn\iris.data')
    train_data, test_data = dataset[:100], dataset[100:]
    train_label, test_label = label[:100], label[100:]
    # pre_label = knn.predict(train_data, train_label, test_data, test_label)
    knn.evaluate(train_data, train_label, test_data, test_label)


if __name__ == "__main__":
    main()
