import math
import random
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


class LogisticRegression():
    def __init__(self):
        self.dataset = None
        self.label = None
        self.weights = None

    def load_dataset(self, path):
        # return arrays of dataset and relevant label
        # when traversal python set,order may be different
        # so chooes list as unique_label's data structure
        dataset, label, unique_label = [], [], []

        # load data from file
        f = open(path)
        for line in f.readlines():
            data = line.strip().split(',')
            # add 'x0=1.0' as first feature,because weight have const 'w0'
            dataset.append([1.0]+data)
            if data[-1] not in unique_label:
                unique_label.append(data[-1])

        # convert each unique label to index
        label2index, index = {}, 0
        for ul in unique_label:
            label2index[ul] = index
            index += 1

        # random.shuffle(dataset) # shuffle the dataset
        dataset = dataset[0: 100]  # top 100 samples,just two class

        # convert label(unique label-->index) in dataset array and generate the label array
        for i in range(len(dataset)):
            label.append(label2index[dataset[i][-1]])
            # convert each string value to float!
            dataset[i] = list(map(float, dataset[i][: -1]))
        self.dataset, self.label = dataset, label
        return dataset, label

    def _gradient_descent(self, alpha, epoch):
        # gradient desscent train function,return trained weights vector
        dataset_mat = np.mat(self.dataset, dtype=float)  # must assign dtype!
        label_mat = np.mat(self.label).transpose()  # 1*m --> m*1

        m, n = np.shape(dataset_mat)  # m*n

        weights = np.ones((n, 1))  # n*1 [[1.],[1.],[1.],[1.]]

        for e in range(epoch):
            # numpy matrix's * operator equal to np.dot
            h = sigmoid(dataset_mat*weights)
            error = label_mat-h
            weights = weights+alpha*dataset_mat.transpose()*error

        weights = np.array(weights)  # matrix convert back to ndarray!
        # shape convert back to 1*n (n,)
        weights = np.reshape(weights, (n)).tolist()
        self.weights = weights
        return weights

    def _stochastic_gradient_descent(self, alpha, epoch):
        # stochastic gradient descent train function,return trained weights vector
        dataset_mat = np.array(self.dataset)

        m, n = np.shape(dataset_mat)

        weights = np.ones(n)  # 1*n [1.,1.,1.,1.]

        for e in range(epoch):
            for i in range(m):
                h = sigmoid(sum(dataset_mat[i]*weights))
                error = self.label[i]-h
                weights = weights+alpha*error*dataset_mat[i]
        weights = weights.tolist()
        self.weights = weights
        return weights

    def train(self, alpha=0.01, epoch=100, optm_algorithm='BGD'):
        # train function,choose optimization algorithm
        if not (self.dataset and self.label):
            print('No dataset or label have been initialized!')
            return

        if optm_algorithm == 'BGD':
            weights = self._gradient_descent(alpha, epoch)
        elif optm_algorithm == 'SGD':
            weights = self._stochastic_gradient_descent(alpha, epoch)
        return weights

    def predict(self, test_data):
        # classfication function,input test data array and weights vector
        if not self.weights:
            print('Model has not been trained!')
            return
        weights = np.array(self.weights)
        # here * means np.multipy,not the np matrix's *
        h = sigmoid(sum(test_data*weights))
        if h > 0.5:
            return 1
        else:
            return 0

    def draw_decision_boundary(self):
        # draw decision boundary from trained model
        # only for two binary classfication!
        if not self.weights:
            print('Model has not been trained!')
            return

        dataset = np.array(self.dataset)
        label, weights = self.label, self.weights
        m = np.shape(dataset)[0]  # dataset's count

        x1, y1, x2, y2 = [], [], [], []

        for i in range(m):
            if label[i] == 0:
                x1.append(dataset[i, 1])
                y1.append(dataset[i, 2])
            else:
                x2.append(dataset[i, 1])
                y2.append(dataset[i, 2])

        fig = plt.figure('decision boundary')
        plt.scatter(x1, y1, s=30, label='class1', c='red')
        plt.scatter(x2, y2, s=30, label='class2', c='green')
        x = np.arange(min(min(x1), min(x2))-5, max(max(y1), max(y2))+5, 0.5)

        y = (-weights[0]-weights[1]*x)/weights[2]
        plt.plot(x, y)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.title('Decision Boundary')
        plt.show()

    def evaluate(self, dataset=None, label=None):
        if not self.weights:
            print('Model has not been trained!')
            return
        if not dataset:
            dataset = self.dataset
        if not label:
            label = self.label
        weights = self.weights

        wrong = 0  # number of test datas that be wrongly classified
        for i in range(len(dataset)):
            y = self.predict(dataset[i])
            if y != label[i]:
                wrong += 1
        accuracy = (1-wrong/len(dataset))*100
        print('Accuracy is %.1f%%' % accuracy)


def main():
    path = 'F:\project\python\ML\logistic_regression\iris.data'
    lg = LogisticRegression()
    lg.load_dataset(path)
    weights = lg.train(alpha=0.005, epoch=10, optm_algorithm='BGD')
    print('weights:', weights)
    lg.draw_decision_boundary()
    lg.evaluate()


if __name__ == "__main__":
    main()
