# 2023-5-3 by Zhaorui.Tan21@xjtlu.edu.cn
# For INT304 Final Assignment
# Here is a numpy version MLP code
# As usual, you need to fill code in format as follows:

#########################################################################################
#                           code you need to fill
# some code/ comments here you need to fill
#########################################################################################

# GOOD LUCK :) !

import numpy as np
import torch.utils.data
import time
import random

random.seed(int(time.time()))
np.random.seed(int(time.time()))

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

random.seed(int(time.time()))
np.random.seed(int(time.time()))




class MLP():
    def __init__(self, train_dl, test_dl, epoch: int, learning_rate: float, gamma=1,
                 initialization="Xavier", gradient_descent_strategy="SGD",
                 data_dim=784, label_dim=10, hidden_nodes=20,):
        # Gradient Descent strategy
        self.gradient_descent_strategy = gradient_descent_strategy

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma  # learning_rate decay hyperparameter gamma
        self.epoch = epoch
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.hidden_nodes = hidden_nodes
        self.initialization = initialization

        # Metrics
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

        # Dataloader
        self.train_dl = train_dl
        self.test_dl = test_dl
        # Inter Variable like z1, a1, z2
        self.inter_variable = {}

        # Gradient Descent Parameter
        self.momentum_v_layer1 = 0
        self.momentum_v_layer2 = 0
        self.momentum_beta = 0.9

        # RMSprop hyperparameters can use larger learning rate
        self.RMS_s_layer1 = 0
        self.RMS_s_layer2 = 0
        self.RMS_beta = 0.999
        self.RMS_epsilon = 1e-8

        # Adam hyperparameters
        self.Adam_v_layer1 = 0
        self.Adam_v_layer2 = 0
        self.Adam_s_layer1 = 0
        self.Adam_s_layer2 = 0
        self.Adam_beta1 = 0.9
        self.Adam_beta2 = 0.999
        self.Adam_epsilon = 1e-8

    def initialize_weights(self):
        if self.initialization == "Xavier":
            #########################################################################################
            #                           code you need to fill
            #  Pay attention to matrix shape in all codes
            w1 = np.random.uniform(low=-math.sqrt(6.0 / float(784 + 20 + 1)),
                                   high=math.sqrt(6.0 / float(784 + 20 + 1)),
                                   size=(784, 20))
            w2 = np.random.uniform(low=-math.sqrt(6.0 / float(20 + 10 + 1)),
                                   high=math.sqrt(6.0 / float(20 + 10 + 1)),
                                   size=(20, 10))
            #########################################################################################
        elif self.initialization == "He":
            w1 = np.random.normal(0, math.sqrt(2 / 20), (784, 20))
            w2 = np.random.normal(0, math.sqrt(2 / 10), (20, 10))
        elif self.initialization == "Gaussian":
            w1 = np.random.normal(0, 1, (784, 20))
            w2 = np.random.normal(0, 1, (20, 10))
        elif self.initialization == "Random":
            w1 = np.random.rand(784, 20) * 2 - 1
            w2 = np.random.rand(20, 10) * 2 - 1
        elif self.initialization == "Constant0":
            w1 = np.zeros((784, 20))
            w2 = np.zeros((20, 10))
        else:
            raise NotImplemented
        return w1, w2

    def train(self, optimizer, activation, gradient_check=False):
        start = time.time()
        w1, w2 = self.initialize_weights()

        for j in range(self.epoch):
            ema_train_accuracy = None
            ema_train_loss = None

            for step, data in enumerate(self.train_dl):
                learning_rate = self.learning_rate
                train_data, train_labels = data
                # 784 * 500
                train_data = train_data.view(train_data.shape[0], -1).numpy().T
                # 500 * 10
                train_labels = F.one_hot(train_labels).numpy()
                if self.gradient_descent_strategy == "SGD":
                        # forward feed
                        self.forward(x=train_data, w1=w1, w2=w2, no_gradient=False, activation=activation)
                        # Calculate gradient
                        gradient1, gradient2 = self.back_prop(x=train_data, y=train_labels, w1=w1, w2=w2,
                                                              activation=activation)
                        w1, w2, learning_rate = self.update_weight(w1, w2, gradient1, gradient2,
                                                                   optimizer=optimizer, epoch=j + 1,
                                                                   learning_rate=learning_rate)
                        train_accuracy = self.accuracy(train_labels, self.inter_variable["z2"])
                        train_loss = self.loss(self.inter_variable["z2"], train_labels)

                        # Gradient check if required
                        if gradient_check:
                            self.gradient_check(train_data, train_labels, w1, w2, gradient1, gradient2,
                                                activation=activation)

                        if ema_train_accuracy is not None:
                            ema_train_accuracy = ema_train_accuracy * 0.98 + train_accuracy * 0.02
                            ema_train_loss = ema_train_loss * 0.98 + train_loss * 0.02

                        else:
                            ema_train_accuracy = train_accuracy
                            ema_train_loss = train_loss
                        if step % 50 == 0:
                            print(f'Train:Step/Epoch:{step}/{j}, Accuracy:{train_accuracy*100:.2f}, Loss:{train_loss:.4f}')
                else:
                    raise NotImplemented

            # Evaluate
            temp_test_accuracy = []
            temp_test_loss = []
            for step, data in enumerate(self.test_dl):
                test_data, test_labels = data
                test_data = test_data.view(test_data.shape[0], -1).numpy().T
                test_labels = F.one_hot(test_labels).numpy()

                test_forward = self.forward(test_data, w1, w2, no_gradient=True, activation=activation)
                test_accuracy = self.accuracy(test_labels, test_forward)
                test_loss = self.loss(test_forward, test_labels)
                temp_test_accuracy.append(test_accuracy)
                temp_test_loss.append(test_loss)

            current_test_accuracy = np.mean(temp_test_accuracy)
            current_test_loss = np.mean(temp_test_loss)
            print(f"Epoch:{j + 1}")
            print(f"Test: Accuracy: {(100 * current_test_accuracy):.2f}%, Loss: {current_test_loss:.4f}")
            # for plot
            self.train_accuracy.append(ema_train_accuracy)
            self.train_loss.append(ema_train_loss)
            self.test_accuracy.append(current_test_accuracy)
            self.test_loss.append(current_test_loss)

        end = time.time()
        print(f"Trained time: {1000 * (end - start)} ms")

    def forward(self, x, w1, w2, no_gradient: bool, activation):
        """
        :param x: Input Data
        :param no_gradient: distinguish it's train or evaluate
        :return: if no_gradient = False, return output
        """

        if activation == "Tanh":
            z1 = w1.T.dot(x)
            a1 = np.tanh(z1)
            z2 = w2.T.dot(a1)
        elif activation == "ReLU":
            z1 = w1.T.dot(x)
            a1 = np.maximum(0, z1)
            z2 = w2.T.dot(a1)
        elif activation == "Sigmoid":
            z1 = w1.T.dot(x)
            a1 = 1 / (1 + np.exp(-z1))
            z2 = w2.T.dot(a1)

        if no_gradient:
            # for predict
            return z2
        else:
            # For back propagation
            self.inter_variable = {"z1": z1, "a1": a1, "z2": z2}

    def back_prop(self, x, y, w1, w2, activation):
        """
        :param i: for Adam bias correction
        """
        m = x.shape[1]

        #########################################################################################
        #                           code you need to fill
        #  Pay attention to matrix shape in all codes
        if activation == "Tanh":
            delta_k = self.inter_variable["z2"] - y.T
            delta_j = (1 - self.inter_variable["a1"] ** 2) * (w2.dot(delta_k))
            gradient1 = 1. / m * (x.dot(delta_j.T))
            gradient2 =  1. / m *  (self.inter_variable["a1"].dot(delta_k.T))
            return gradient1, gradient2
        elif activation == "Sigmoid":
            delta_k =  self.inter_variable["z2"] - y.T
            delta_j =  self.inter_variable["a1"] * (1 - self.inter_variable["a1"]) * (w2.dot(delta_k))
            gradient1 = 1. / m * (x.dot(delta_j.T))
            gradient2 = 1. / m * (self.inter_variable["a1"].dot(delta_k.T))
            return gradient1, gradient2
        elif activation == "ReLU":
            delta_k = self.inter_variable["z2"] - y.T
            delta_relu = self.inter_variable["a1"]
            delta_relu[delta_relu <= 0] = 0
            delta_relu[delta_relu > 0] = 1
            delta_j = delta_relu * (w2.dot(delta_k))
            gradient1 = 1. / m * (x.dot(delta_j.T))
            gradient2 = 1. / m * (self.inter_variable["a1"].dot(delta_k.T))
            return gradient1, gradient2
        #########################################################################################

    def update_weight(self, w1, w2, gradient1, gradient2, optimizer, epoch, learning_rate):
        if optimizer == "SGD":
            return self.SGD(w1, w2, gradient1, gradient2, learning_rate)
        elif optimizer == "Momentum":
            return self.Momentum(w1, w2, gradient1, gradient2, learning_rate)
        elif optimizer == "RMSprop":
            return self.RMSprop(w1, w2, gradient1, gradient2, learning_rate)
        elif optimizer == "Adam":
            return self.Adam(epoch, w1, w2, gradient1, gradient2, learning_rate)

    def SGD(self, w1, w2, gradient1, gradient2, learning_rate):
        w1 = w1 - learning_rate * gradient1
        w2 = w2 - learning_rate * gradient2
        # Learning rate decay
        learning_rate *= self.gamma
        return w1, w2, learning_rate

    def Momentum(self, w1, w2, gradient1, gradient2, learning_rate):
        """Exponential weighted average"""
        #  you can do it if you want
        pass

    def RMSprop(self, w1, w2, gradient1, gradient2, learning_rate):
        """Mean squared prop"""
        #  you can do it if you want
        pass

    def Adam(self, t, w1, w2, gradient1, gradient2, learning_rate):
        """Adaption moment estimation"""
        #  you can do it if you want
        pass

    @staticmethod
    def accuracy(label, y_hat: np.ndarray):
        y_hat = y_hat.T
        acc = y_hat.argmax(axis=1) == label.argmax(axis=1)
        b = acc + 0
        return b.mean()

    def save(self, filename):
        np.savez(filename, self.weights1_list, self.weights2_list)

    @staticmethod
    def loss(output, label):
        # Loss = 1/n * 1/2 * ∑(yk - tk)^2
        a = label.shape[0]
        return np.sum(((output.T - label) ** 2)) / (2 * label.shape[0])

    def gradient_check(self, x, y, w1, w2, gradient1, gradient2, activation, epsilon=1e-7):
        parameters = np.vstack((w1.reshape((100, 1)), w2.reshape((60, 1))))
        grad = np.vstack((gradient1.reshape((100, 1)), gradient2.reshape(60, 1)))
        num_parameters = parameters.shape[0]
        gradapprox = np.zeros((num_parameters, 1))
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        for i in range(num_parameters):
            thetaplus = np.copy(parameters)
            thetaplus[i][0] = thetaplus[i][0] + epsilon
            w_plus_layer1 = thetaplus[0:100].reshape(5, 20)
            w_plus_layer2 = thetaplus[100:160].reshape(20, 3)
            J_plus[i] = self.evaluate(x, y, w_plus_layer1, w_plus_layer2, activation)

            thetaminus = np.copy(parameters)
            thetaminus[i][0] = thetaminus[i][0] - epsilon
            w_minus_layer1 = thetaminus[0:100].reshape(5, 20)
            w_minus_layer2 = thetaminus[100:160].reshape(20, 3)
            J_minus[i] = self.evaluate(x, y, w_minus_layer1, w_minus_layer2, activation)
            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2. * epsilon)
        numerator = np.linalg.norm(grad - gradapprox)
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
        difference = numerator / denominator
        print(f"L2 distance of Gradient check:{difference}")

    def evaluate(self, x, y, w1, w2, activation):
        z1 = w1.T.dot(x)
        if activation == "Tanh":
            a1 = np.tanh(z1)
        elif activation == "ReLU":
            a1 = np.maximum(0, z1)
        elif activation == "Sigmoid":
            a1 = 1 / (1 + np.exp(-z1))
        z2 = w2.T.dot(a1)
        return np.sum(((z2.T - y) ** 2) / (2 * y.shape[0]))


    def plot_test(self):
        plt.figure(figsize=(7, 6))
        plt.xlabel(f"Epochs({self.epoch} Epoch)")
        plt.ylabel("Accuracy")
        plt.plot(self.test_accuracy, label="Test Accuracy", alpha=0.5)
        plt.xticks(np.arange(0, len(self.test_accuracy)) )
        plt.legend()
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(7, 6))
        plt.xlabel("Epochs(10 Epoch/step)")
        plt.ylabel("Accuracy")
        plt.plot(np.array(self.train_loss), label="Train Loss", alpha=0.5)
        plt.xticks(np.arange(0, len(self.train_loss)))
        plt.legend()
        plt.show()



if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    # 500 * 1 * 28 * 28
    train_dl = torch.utils.data.DataLoader(train_set, batch_size=500, drop_last=True, num_workers=4, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_set, batch_size=500, drop_last=False, num_workers=4, shuffle=True)
    # you need to tune different parameters here

    #########################################################################################
    #                           code you need to fill
    # pass different hyper parameters into the model
    mlp = MLP(train_dl=train_dl, test_dl=test_dl, epoch=10, learning_rate=0.01, gamma=0.5, initialization="Xavier",
              gradient_descent_strategy="SGD", data_dim=784, label_dim=10, hidden_nodes= 20)
    #########################################################################################

    mlp.train(optimizer="SGD", activation="Tanh", gradient_check=False)
    mlp.plot_test()
    mlp.plot_loss()
