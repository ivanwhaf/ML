import random
import numpy as np


def sigmoid(x):
    # sigmoid function: f(x) = 1 / (1 + e^(-x))
    return 1/(1+np.exp(-x))


def deriv_sigmoid(x):
    # derivative of simgoid function: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx*(1-fx)


def mse_loss(y_true, y_predict):
    # loss function
    return ((y_true-y_predict)**2).mean()


def get_random_weights(size) -> list:
    # get random weight value from -1 to 1
    ret = []
    for i in range(size):
        ret.append(random.uniform(-1, 1))
    return ret


def get_random_bias():
    # get random bias value from -1 to 1
    return random.uniform(-1, 1)


class Neuron:
    """
    single neuron class
    """

    def __init__(self, weights, bias, activation=sigmoid):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def feedforward(self, inputs):
        '''
        output=0
        for i in range(len(inputs)):
            output+=inputs[i]*weights[i]
        '''
        output = np.dot(self.weights, inputs)+self.bias
        return self.activation(output)


class NeuralNetwork:
    """
    this network consist of three layers including 1 input layer, 1 hidden layer, and 1 output layer
    input layer has 3 neurons, hidden layer has 5 neurons, and output layer has 2 neurons
    ┌─────────────┬──────────────┬──────────────┐
     input layer  hidden layer  output layer 
    ├─────────────┼──────────────┼──────────────┤
          3             5            2            
    └─────────────┴──────────────┴──────────────┘
    """

    def __init__(self):
        # hidden layer neurons's initial weights
        hidden_weights = [0, 1, 0]
        # output layer neurons's initial weights
        output_weights = [0, 1, 0, 1, 0]
        # all neurons's initial bias: 0
        bias = 1

        # hidden layer neurons
        self.h1 = Neuron(get_random_weights(3), get_random_bias())
        self.h2 = Neuron(get_random_weights(3), get_random_bias())
        self.h3 = Neuron(get_random_weights(3), get_random_bias())
        self.h4 = Neuron(get_random_weights(3), get_random_bias())
        self.h5 = Neuron(get_random_weights(3), get_random_bias())
        self.hs = [self.h1, self.h2, self.h3, self.h4, self.h5]

        # output layer neurons
        self.o1 = Neuron(get_random_weights(5), get_random_bias())
        self.o2 = Neuron(get_random_weights(5), get_random_bias())
        self.os = [self.o1, self.o2]

    def feedforward(self, inputs):
        output_h1 = self.h1.feedforward(inputs)
        output_h2 = self.h2.feedforward(inputs)
        output_h3 = self.h3.feedforward(inputs)
        output_h4 = self.h4.feedforward(inputs)
        output_h5 = self.h5.feedforward(inputs)

        output_h = (output_h1, output_h2, output_h3, output_h4, output_h5)

        output_o1 = self.o1.feedforward(output_h)
        output_o2 = self.o2.feedforward(output_h)

        return (output_o1, output_o2)

    def train(self, x_train, y_train, lr=0.01, epochs=100):
        # train loop times
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                # feedforward
                sum_h1 = self.h1.weights[0]*x[0] + self.h1.weights[2] * \
                    x[1] + self.h1.weights[2]*x[2] + self.h1.bias
                h1 = sigmoid(sum_h1)

                sum_h2 = self.h2.weights[0]*x[0] + self.h2.weights[2] * \
                    x[1] + self.h2.weights[2]*x[2] + self.h2.bias
                h2 = sigmoid(sum_h2)

                sum_h3 = self.h3.weights[0]*x[0] + self.h3.weights[2] * \
                    x[1] + self.h3.weights[2]*x[2] + self.h3.bias
                h3 = sigmoid(sum_h3)

                sum_h4 = self.h4.weights[0]*x[0] + self.h4.weights[2] * \
                    x[1] + self.h4.weights[2]*x[2] + self.h4.bias
                h4 = sigmoid(sum_h4)

                sum_h5 = self.h5.weights[0]*x[0] + self.h5.weights[2] * \
                    x[1] + self.h5.weights[2]*x[2] + self.h5.bias
                h5 = sigmoid(sum_h5)

                sum_o1 = self.o1.weights[0]*h1 + \
                    self.o1.weights[1]*h2+self.o1.weights[2]*h3
                + self.o1.weights[3]*h4+self.o1.weights[4]*h5+self.o1.bias
                o1 = sigmoid(sum_o1)

                sum_o2 = self.o2.weights[0]*h1 + \
                    self.o2.weights[1]*h2+self.o2.weights[2]*h3
                + self.o2.weights[3]*h4+self.o2.weights[4]*h5+self.o2.bias
                o2 = sigmoid(sum_o2)

                # ---------back propagation---------
                # L=(y_true-y_pred)²
                # dL/dy
                dL_dy1 = -2*(y[0]-o1)
                dL_dy2 = -2*(y[1]-o2)

                # y=f(wh+b) output layer
                # dy/dw dy/db
                dy1_dw1 = h1*deriv_sigmoid(sum_o1)
                dy1_dw2 = h2*deriv_sigmoid(sum_o1)
                dy1_dw3 = h3*deriv_sigmoid(sum_o1)
                dy1_dw4 = h4*deriv_sigmoid(sum_o1)
                dy1_dw5 = h5*deriv_sigmoid(sum_o1)
                dy1_db = deriv_sigmoid(sum_o1)

                dy2_dw1 = h1*deriv_sigmoid(sum_o2)
                dy2_dw2 = h2*deriv_sigmoid(sum_o2)
                dy2_dw3 = h3*deriv_sigmoid(sum_o2)
                dy2_dw4 = h4*deriv_sigmoid(sum_o2)
                dy2_dw5 = h5*deriv_sigmoid(sum_o2)
                dy2_db = deriv_sigmoid(sum_o2)

                # y=f(wh+b) output layer
                # dy/dh
                dy1_dh1 = self.o1.weights[0]*deriv_sigmoid(sum_o1)
                dy1_dh2 = self.o1.weights[1]*deriv_sigmoid(sum_o1)
                dy1_dh3 = self.o1.weights[2]*deriv_sigmoid(sum_o1)
                dy1_dh4 = self.o1.weights[3]*deriv_sigmoid(sum_o1)
                dy1_dh5 = self.o1.weights[4]*deriv_sigmoid(sum_o1)

                dy2_dh1 = self.o2.weights[0]*deriv_sigmoid(sum_o2)
                dy2_dh2 = self.o2.weights[1]*deriv_sigmoid(sum_o2)
                dy2_dh3 = self.o2.weights[2]*deriv_sigmoid(sum_o2)
                dy2_dh4 = self.o2.weights[3]*deriv_sigmoid(sum_o2)
                dy2_dh5 = self.o2.weights[4]*deriv_sigmoid(sum_o2)

                # h=f(wx+b) hidden layer
                # dh/dw dw/db
                dh1_dw1 = x[0]*deriv_sigmoid(sum_h1)
                dh1_dw2 = x[1]*deriv_sigmoid(sum_h1)
                dh1_dw3 = x[2]*deriv_sigmoid(sum_h1)
                dh1_db = deriv_sigmoid(sum_h1)

                dh2_dw1 = x[0]*deriv_sigmoid(sum_h2)
                dh2_dw2 = x[1]*deriv_sigmoid(sum_h2)
                dh2_dw3 = x[2]*deriv_sigmoid(sum_h2)
                dh2_db = deriv_sigmoid(sum_h2)

                dh3_dw1 = x[0]*deriv_sigmoid(sum_h3)
                dh3_dw2 = x[1]*deriv_sigmoid(sum_h3)
                dh3_dw3 = x[2]*deriv_sigmoid(sum_h3)
                dh3_db = deriv_sigmoid(sum_h3)

                dh4_dw1 = x[0]*deriv_sigmoid(sum_h4)
                dh4_dw2 = x[1]*deriv_sigmoid(sum_h4)
                dh4_dw3 = x[2]*deriv_sigmoid(sum_h4)
                dh4_db = deriv_sigmoid(sum_h4)

                dh5_dw1 = x[0]*deriv_sigmoid(sum_h5)
                dh5_dw2 = x[1]*deriv_sigmoid(sum_h5)
                dh5_dw3 = x[2]*deriv_sigmoid(sum_h5)
                dh5_db = deriv_sigmoid(sum_h5)

                # update weights and bias
                # output layer
                self.o1.weights[0] -= lr*dL_dy1*dy1_dw1
                self.o1.weights[1] -= lr*dL_dy1*dy1_dw2
                self.o1.weights[2] -= lr*dL_dy1*dy1_dw3
                self.o1.weights[3] -= lr*dL_dy1*dy1_dw4
                self.o1.weights[4] -= lr*dL_dy1*dy1_dw5
                self.o1.bias -= lr*dL_dy1*dy1_db

                self.o2.weights[0] -= lr*dL_dy2*dy2_dw1
                self.o2.weights[1] -= lr*dL_dy2*dy2_dw2
                self.o2.weights[2] -= lr*dL_dy2*dy2_dw3
                self.o2.weights[3] -= lr*dL_dy2*dy2_dw4
                self.o2.weights[4] -= lr*dL_dy2*dy2_dw5
                self.o2.bias -= lr*dL_dy2*dy2_db

                # hidden layer
                # ------y1------
                self.h1.weights[0] -= lr*dL_dy1*dy1_dh1*dh1_dw1
                self.h1.weights[1] -= lr*dL_dy1*dy1_dh1*dh1_dw2
                self.h1.weights[2] -= lr*dL_dy1*dy1_dh1*dh1_dw3
                self.h1.bias -= lr*dL_dy1*dy1_dh1*dh1_db

                self.h2.weights[0] -= lr*dL_dy1*dy1_dh2*dh2_dw1
                self.h2.weights[1] -= lr*dL_dy1*dy1_dh2*dh2_dw2
                self.h2.weights[2] -= lr*dL_dy1*dy1_dh2*dh2_dw3
                self.h2.bias -= lr*dL_dy1*dy1_dh2*dh2_db

                self.h3.weights[0] -= lr*dL_dy1*dy1_dh3*dh3_dw1
                self.h3.weights[1] -= lr*dL_dy1*dy1_dh3*dh3_dw2
                self.h3.weights[2] -= lr*dL_dy1*dy1_dh3*dh3_dw3
                self.h3.bias -= lr*dL_dy1*dy1_dh3*dh3_db

                self.h4.weights[0] -= lr*dL_dy1*dy1_dh4*dh4_dw1
                self.h4.weights[1] -= lr*dL_dy1*dy1_dh4*dh4_dw2
                self.h4.weights[2] -= lr*dL_dy1*dy1_dh4*dh4_dw3
                self.h4.bias -= lr*dL_dy1*dy1_dh4*dh4_db

                self.h5.weights[0] -= lr*dL_dy1*dy1_dh5*dh5_dw1
                self.h5.weights[1] -= lr*dL_dy1*dy1_dh5*dh5_dw2
                self.h5.weights[2] -= lr*dL_dy1*dy1_dh5*dh5_dw3
                self.h5.bias -= lr*dL_dy1*dy1_dh5*dh5_db

                # -----y2---------
                self.h1.weights[0] -= lr*dL_dy2*dy2_dh1*dh1_dw1
                self.h1.weights[1] -= lr*dL_dy2*dy2_dh1*dh1_dw2
                self.h1.weights[2] -= lr*dL_dy2*dy2_dh1*dh1_dw3
                self.h1.bias -= lr*dL_dy2*dy2_dh1*dh1_db

                self.h2.weights[0] -= lr*dL_dy2*dy2_dh2*dh2_dw1
                self.h2.weights[1] -= lr*dL_dy2*dy2_dh2*dh2_dw2
                self.h2.weights[2] -= lr*dL_dy2*dy2_dh2*dh2_dw3
                self.h2.bias -= lr*dL_dy2*dy2_dh2*dh2_db

                self.h3.weights[0] -= lr*dL_dy2*dy2_dh3*dh3_dw1
                self.h3.weights[1] -= lr*dL_dy2*dy2_dh3*dh3_dw2
                self.h3.weights[2] -= lr*dL_dy2*dy2_dh3*dh3_dw3
                self.h3.bias -= lr*dL_dy2*dy2_dh3*dh3_db

                self.h4.weights[0] -= lr*dL_dy2*dy2_dh4*dh4_dw1
                self.h4.weights[1] -= lr*dL_dy2*dy2_dh4*dh4_dw2
                self.h4.weights[2] -= lr*dL_dy2*dy2_dh4*dh4_dw3
                self.h4.bias -= lr*dL_dy2*dy2_dh4*dh4_db

                self.h5.weights[0] -= lr*dL_dy2*dy2_dh5*dh5_dw1
                self.h5.weights[1] -= lr*dL_dy2*dy2_dh5*dh5_dw2
                self.h5.weights[2] -= lr*dL_dy2*dy2_dh5*dh5_dw3
                self.h5.bias -= lr*dL_dy2*dy2_dh5*dh5_db

            if (epoch+1) % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, x_train)
                loss = mse_loss(y_train, y_preds)
                print("Epoch: %d,loss: %.3f" % (epoch+1, loss))


def main():
    weights, bias = [1, 2], 5
    neuron = Neuron(weights, bias)
    inputs = [2, 3]
    print(neuron.feedforward(inputs))

    network = NeuralNetwork()
    inputs = [2, 3, 4]
    print(network.feedforward(inputs))
    x_train = [[1, 2, 3], [10, 3, 4], [0.5, 2.5, 3.5], [15, 3, 5]]
    y_train = [[0, 0], [1, 1], [0, 0], [1, 1]]
    network.train(x_train, y_train, lr=0.1, epochs=500)
    print(network.o1.weights)
    print(network.o2.weights)
    print(network.feedforward([1, 2, 3]))
    print(network.feedforward([12, 4, 5]))


if __name__ == "__main__":
    main()
