import numpy as np
import matplotlib.pyplot as plt


class LinearRegression():
    def __init__(self):
        self.weights = None
        self.dataset = None
        self.label = None

    def load_dataset(self, path=None):
        dataset = [[1.0, 0.5], [1.0, 1], [1.0, 2], [1.0, 3], [1.0, 4], [1.0, 4.5], [1.0, 5], [
            1.0, 6], [1.0, 7], [1.0, 7.5], [1.0, 8], [1.0, 8.5], [1.0, 9], [1.0, 9.5], [1.0, 10]]
        label = [0.3, 1, 1, 2, 1.5, 2, 3, 3.5, 3.7, 4.1, 3.4, 5, 4.2, 4, 5.5]
        self.dataset, self.label = dataset, label
        return dataset, label

    def train(self):
        # Least squares optimization function
        if not (self.dataset and self.label):
            print('No dataset or label have been initialized!')
            return

        x_mat = np.mat(self.dataset)
        y_mat = np.mat(self.label).T

        xTx = x_mat.T*x_mat

        # calculate the matrix's determinant
        if np.linalg.det(xTx) == 0.0:
            print('Matrix is singular,cannot do inverse!')
            return

        weights = xTx.I*(x_mat.T*y_mat)  # weights matrix type

        weights = np.array(weights)  # matrix convert back to ndarray!
        # shape convert back to 1*n (n,)
        weights = np.reshape(weights, (np.shape(x_mat)[1])).tolist()

        self.weights = weights
        return weights

    def draw_fitting_line(self):
        if not self.weights:
            print('Model has not been trained!')
            return

        x_mat = np.mat(self.dataset)
        y_mat = np.mat(self.label)
        y_hat = x_mat*np.mat(self.weights).T

        fig = plt.figure('Best fit grapg')
        plt.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])
        plt.plot(x_mat[:, 1], y_hat)
        for x, y in zip(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0]):
            plt.text(x, y, y, ha='center', fontsize=10, style='italic')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Best fit graph')
        plt.show()


def main():
    lg = LinearRegression()
    lg.load_dataset()
    weights = lg.train()
    print('weights:', weights)
    lg.draw_fitting_line()


if __name__ == "__main__":
    main()
