import numpy as np
import random


class Regclass():
    def __init__(self, learningRate=0.1, n_epochs=2000, rtol=0.01,
                 batch_size="auto",penalty=None):
        if batch_size == "None":
            self.batch_size = lambda n_inputs: n_inputs
        elif batch_size == "auto":
            self.batch_size = lambda n_inputs: min(200, n_inputs)
        elif isinstance(batch_size, int):
            self.batch_size = lambda n_inputs: batch_size
        else:
            raise ValueError("Wrong input!")

        self.learningRate = learningRate
        self.n_epochs = n_epochs
        self.rtol = rtol
        self.penalty = penalty

    def fit(self, X, y):
        pass

    def accuracy(self, X, y):
        score = np.mean(self.predict(X) == y)
        return score


class LogisticRegression(Regclass):
    def fit(self, X, y):
        beta = random.normal(0, np.sqrt(2/X.shape[1]), size=X.shape[1])
        self.gradientdescent(X, y, beta)
        self.beta = beta

    def predict(self, X):
        pred = X @ self.beta
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

    def sigmoid(self, X, beta):
        factor = np.exp(X @ beta)
        P = factor/(1+factor)
        return P

    def cost_func(self, X, y, beta):
        return -X.T @ (y - self.sigmoid(X, beta))

    def gradientdescent(self, X, y):







