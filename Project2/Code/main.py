from __future__ import division
import scipy.stats
import numpy as np
import scipy.special as sps

from numba import njit
from sklearn.base import BaseEstimator, ClassifierMixin


class RegClass(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.1, n_epochs=2000, rtol=0.01,
                 batch_size="auto", verbose=False, learning_schedule=None):
        """
        Base class for the logistic regressor and the multilayer perceptron regressor
        and classifier. Have to be run with another class.
        learning rate: float, default 0.1 - Used in the stochastic gradient
                descent solver.
        n_epochs: int, default 2000 - Number of epochs used in stochastic
                gradient descent solver.
        rtol: float, default 0.01 - Relative tolerance used as a stopping
                criteria in the stochastic gradient descent solver.
        batch_size: int, default "auto" - Size of the minibatches used in the
                stochastic gradient descent solver.
        verbose: bool, default False - If True, then print progress in terminal
        learning_schedule: float, default None - Deprecated
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.rtol = rtol
        self.verbose = verbose
        self.learning_schedule = learning_schedule

    def fit(self, X=None, y=None):
        """
        Raise error if method is run directly.
        """
        raise RuntimeError("Do not use this directly.")

    def get_batch_size(self, num_samp):
        """
        Return the batch size of the minibatches.
        num_samp: Number of data samples
        """
        if self.batch_size == "auto":
            return np.min([200, num_samp])
        elif self.batch_size == None:
            return num_samp
        elif isinstance(self.batch_size, int):
            return self.batch_size
        elif callable(self.batch_size):
            return self.batch_size(num_samp)
        else:
            raise ValueError(f"Only batch size 'auto', 'none', function or \
                             integer supported. batch_size={self.batch_size}")

    def accuracy_score(self, X, y):
        """
        Raises error if method is run directly on its own before a regression
        method.
        """
        if len(y.shape) == 1:
            raise ValueError("y must have shape (n,1), need to reshape")
        with np.errstate(invalid="raise"):
            return np.mean(self.predict(X) == np.array(y, dtype=np.int))


class LogReg(RegClass):
    """
    Performs logistic regression for design matrix X and binary output y.
    """
    def fit(self, X, y):
        """
        Fits the model to the data.
        X: ndarray - Design matrix
        y: array - Binary output
        """
        if len(y.shape) == 1:
            raise ValueError("y must have shape (n,1), need to reshape")
        self.beta = np.random.normal(0, np.sqrt(2/X.shape[1]),
                                     size=X.shape[1]).reshape(-1, 1)
        self.stochastic_GD(X, y)

    def predict_prob(self, X):
        """
        Predicts output probabilities.
        X: ndarray - Design matrix
        """
        exponent = np.exp(X @ self.beta)
        p = exponent/ (1 + exponent)
        return p.reshape(-1, 1)

    def predict(self, X):
        """
        Converts output probabilities to binary output.
        X: ndarray - Design matrix
        """
        pred = self.predict_prob(X)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

    def stochastic_GD(self, X, y):
        """
        Runs stochastic gradient descent for optimization of the cost function.
        Use the cross-entropy as the cost function.
        X: ndarray - Design matrix
        y: array - True binary output
        """
        if self.learning_schedule == None:
            reduce_i = self.n_epochs + 1
        else:
            reduce_i = self.learning_schedule
        n_iterations = len(y) // self.get_batch_size(len(y))
        cost = np.zeros(self.n_epochs)
        y_pred = self.predict(X)
        if self.verbose:
            print(f"Initial cost func: {self.cost_func(y, y_pred):g}")
        for i in range(self.n_epochs):
            if np.any(np.isnan(self.beta)):
                raise ValueError("Invalid value in beta")
            if i % reduce_i == 0 and not i == 0:
                self.learning_rate /= 2
                if self.verbose:
                    print(f"Learning rate reduced to {self.learning_rate}")
            batch_indices = np.array_split(np.random.permutation(len(y)), n_iterations)
            for j in range(n_iterations):
                random_batch = np.random.randint(n_iterations)
                grad = self.grad_cost_function(self.beta,
                                               X[batch_indices[random_batch]],
                                               y[batch_indices[random_batch]],
                                               ).reshape(-1, 1)
                if np.any(np.isnan(grad)):
                    if self.verbose:
                        print(f"NaN in gradient, stopping at epoch {i}")
                    return
                self.beta -= self.learning_rate*grad
            y_pred = self.predict_prob(X)
            cost[i] = self.cost_func(y, y_pred)
            if self.verbose:
                print(f"Epochs {i/self.n_epochs*100:.2f}% done. Cost func: {cost[i]:g}")
            if i > 10:
                cost_diff = (cost[i-11:i] - cost[i-10:i+1])/cost[i-11:i]
                if np.max(cost_diff) < self.rtol:
                    if self.verbose:
                        print(f"Loss function did not improve more than given \
                              relative tolerance {self.rtol:g} for 10 \
                              consecutive epochs (max improvement was \
                              {np.max(cost_diff)}). Stopping at epoch {i:g}")
                    break

    def save_model(self, filename):
        """
        Save the model
        filename: Output file name
        """
        np.savez(f"Models/{filename}", beta=self.beta)

    def load_model(self, filename):
        """
        Load a saved model
        filename: File name
        """
        model = np.load(f"Models/{filename}", allow_pickle=True)
        self.beta = model["beta"].reshape(-1, 1)

    @staticmethod
    def cost_func(y, y_pred):
        """
        Returns the values of the cost function
        y: ndarray - The true binary outputs
        y_pred: ndarray - The predicted binary outputs found using the model
        """
        C = -np.sum(sps.xlogy(y, y_pred) + sps.xlogy(1 - y, 1 - y_pred))/y_pred.shape[0]
        return C

    @staticmethod
    @njit
    def grad_cost_function(beta, X, y):
        """
        Returns the gradient of the cost function
        beta: ndarray - Logistic regression predictor
        X: ndarray - Design matrix
        y: ndarray - True binary output
        """
        exponent = np.exp(X @ beta)
        p = exponent/(1 + exponent)
        return (-X.T @ (y - p)).sum(axis=1)


class MLPClassifier(RegClass):
    def __init__(self, hidden_layer_size=(20,10,5,3), learning_rate=0.1,
                 lambd=0, n_epochs=2000, rtol=0.001, batch_size="auto",
                 verbose=False, activation_function_output="sigmoid",
                 learning_schedule=None):
        """
        Performs classification analysis using a multilayer perceptron
        hidden_layer_size: iterable, default [20, 10, 5, 3] - Number of nodes
                in each hidden layer.
        lambd: float, default 0 - The L2 shrinkage parameter used for
                regularisation of the weights.
        learning rate: float, default 0.1 - The learning rate used in the
                stochastic gradient descent solver.
        n_epochs: int, default 2000 - Number of epochs used in the stochastic
                gradient descent solver.
        rtol: float, default 0.001 - Relative tolerance used as a stopping
                criteria in the stochastic gradient descent solver.
        batch_size: int, default "auto" - Size of the minibatches used in the
                stochastic gradient descent solver.
        verbose: bool, default False - If True, print progress in terminal.
        activation_function_output: string, default "sigmoid"
            Specifies which activation function to use for the output layer.
            Supports "linear" and "sigmoid".
        learning_schedule: float, default None - Deprecated.
        """
        super().__init__(learning_rate, n_epochs, rtol, batch_size, verbose,
                         learning_schedule)
        self.lambd = lambd
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers = len(hidden_layer_size)
        self.activation_function_output = activation_function_output

    def fit(self, X, y):
        """
        Fits the model to the data.
        X: ndarray - Design matrix
        y: array - Binary output
        """
        self.n_features = len(X[0,:])
        self.n_inputs = len(X[:,0])
        if len(y.shape) == 1:
            raise ValueError("y must have shape (n,1), need to reshape")
        else:
            self.n_outputs = y.shape[1]

        self.init_biases_weights()
        self.stochastic_GD(X, y)

    def predict_prob(self, X):
        """
        Predicts output probabilities
        X: ndarray - Design matrix
        """
        if self.weights_hidden[0].shape[0] != X.shape[1]:
            print(len(self.weights_hidden[0].shape[0]), X.shape[1])
            raise ValueError(
                "Model was fitted on different inputs than what was provided"
            )
        return self.feed_forward(X)[0][-1]

    def predict(self, X):
        """
        Converts output probabilities to binary output
        X: ndarray - Design matrix
        """
        if self.weights_hidden[0].shape[0] != X.shape[1]:
            print(len(self.weights_hidden[0].shape[0]), X.shape[1])
            raise ValueError("Model was fitted on different inputs than what was provided")

        pred = self.predict_prob(X)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return np.array(pred, dtype=np.int)

    def accuracy_score(self, X, y):
        """
        Returns accuracy score of the model
        X: ndarray - Design matrix
        y: array - True binary output
        """
        if self.weights_out.shape[1] != y.shape[1]:
            print(self.weights_out.shape[1], y.shape[1])
            raise ValueError("Model was fitted on different outputs than what was provided")
        return super().accuracy_score(X, y)

    def init_biases_weights(self):
        """
        Initializes the biases and the weights.
        """
        std_weight_init = np.sqrt(1/self.n_features)

        self.weights_hidden = []
        self.biases_hidden = []

        for i in range(self.n_hidden_layers):
            if i == 0:
                hidden_weights = np.random.uniform(-std_weight_init, std_weight_init,
                                size=(self.n_features, self.hidden_layer_size[i]))
            else:
                hidden_weights = np.random.uniform(-std_weight_init, std_weight_init,
                    size=(self.hidden_layer_size[i-1], self.hidden_layer_size[i]))

            hidden_biases = np.zeros(self.hidden_layer_size[i]) + 0.01

            self.weights_hidden.append(hidden_weights)
            self.biases_hidden.append(hidden_biases)

        self.weights_out = np.random.normal(loc=0, scale=std_weight_init,
                            size=(self.hidden_layer_size[-1], self.n_outputs))
        self.biases_out = np.zeros(self.n_outputs) + 0.01

    def feed_forward(self, X):
        """
        Feed-forward function.
        X: ndarray - Design matrix
        """
        a_i = np.zeros(self.n_hidden_layers + 2, dtype=np.ndarray)
        z_i = np.zeros(self.n_hidden_layers + 2, dtype=np.ndarray)

        a_i[0] = X
        z_i[0] = np.array([0])

        for i in range(self.n_hidden_layers):
            z_i[i+1] = a_i[i] @ self.weights_hidden[i] + self.biases_hidden[i]
            a_i[i+1] = self.sigmoid(z_i[i+1])

        z_i[-1] = a_i[-2] @ self.weights_out + self.biases_out
        a_i[-1] = self.activation_function_out(z_i[-1], self.activation_function_output)
        return a_i, z_i

    def backpropagation(self, X, y):
        """
        Backpropagation function. Iterates through all epochs or until a given
        tolerance is reached.
        """
        a_i, z_i = self.feed_forward(X)
        delta = np.zeros(self.n_hidden_layers + 1, dtype=np.ndarray)
        gradient_bias = np.zeros_like(delta)
        gradient_weight = np.zeros_like(delta)

        delta[-1] = self.grad_cost(y, a_i[-1])*self.grad_activation_out(z_i[-1],
                     self.activation_function_output)
        gradient_bias[-1] = np.sum(delta[-1], axis=0)
        gradient_weight[-1] = (delta[-1].T @ a_i[-2]).T

        delta[-2] = self.weights_out @ delta[-1].T*self.grad_activation(z_i[-2]).T
        gradient_bias[-2] = np.sum(delta[-2], axis=1)
        gradient_weight[-2] = delta[-2] @ a_i[-3]

        for l in range(-3, -self.n_hidden_layers - 2, -1):
            delta[l] = (self.weights_hidden[l+2] @ delta[l+1]
                        *self.grad_activation(z_i[l]).T)
            gradient_bias[l] = np.sum(delta[l], axis=1)
            gradient_weight[l] = delta[l] @ a_i[l-1]
        return gradient_weight, gradient_bias

    def stochastic_GD(self, X, y):
        """
        Run stochastic gradient descent for optimization of the cost function.
        X: ndarray - Design matrix
        y: array - True binary output
        """
        if self.learning_schedule == None:
            reduce_i = self.n_epochs + 1
        else:
            reduce_i = self.learning_schedule
        n_iterations = len(y) // self.get_batch_size(len(y))
        cost = np.zeros(self.n_epochs)
        y_pred = self.feed_forward(X)[0][-1]
        lambd_feat = self.lambd/self.n_features
        if self.verbose:
            print(f"Initial cost func: {self.cost_func(y,y_pred):g}")

        for i in range(self.n_epochs):
            if i % reduce_i == 0 and not i == 0:
                self.learning_rate /= 2
                if self.verbose:
                    print(f"Learning rate reduced to {self.learning_rate}")
            batch_indices = np.array_split(np.random.permutation(len(y)), n_iterations)
            for j in range(n_iterations):
                random_batch = np.random.randint(n_iterations)
                gradients_weight, gradients_bias = self.backpropagation(
                    X[batch_indices[random_batch]], y[batch_indices[random_batch]])
                if np.any(np.isnan(gradients_weight[-1])) or np.any(np.isnan(gradients_bias[-1])):
                    if self.verbose:
                        print(f"NaN gradient detected, stopping at epoch {i}.")
                    return
                # Output layer
                self.weights_out -= (self.learning_rate*gradients_weight[-1]
                                    + self.weights_out*lambd_feat)
                self.biases_out -= self.learning_rate*gradients_bias[-1]
                # Hidden layer
                for l in range(-1, -self.n_hidden_layers - 1, -1):
                    if np.any(np.isnan(gradients_weight[l])) or np.any(np.isnan(gradients_bias[l])):
                        if self.verbose:
                            print(f"NaN gradient detected, stopping at epoch {i}.")
                        return
                    self.weights_hidden[l] -= (self.learning_rate*gradients_weight[l-1].T
                                           + self.weights_hidden[l]*lambd_feat)
                    self.biases_hidden[l] -= self.learning_rate*gradients_bias[l-1]
            y_pred = self.feed_forward(X)[0][-1]
            cost[i] = self.cost_func(y, y_pred)
            if self.verbose:
                print(f"Epochs {i/self.n_epochs*100:.2f}% done. Cost func: {cost[i]:g}")
            if i > 10:
                cost_diff = (cost[i-11:i] - cost[i-10:i+1])/cost[i-11:i]
                if np.max(cost_diff) < self.rtol:
                    if self.verbose:
                        print(f"Loss function did not improve more than given \
                              relative tolerance {self.rtol:g} for 10 \
                              consecutive epochs (max improvement was \
                              {np.max(cost_diff)}). Stopping at epoch {i:g}")
                    break

    def save_model(self, filename):
        """
        Saves the weights and biases.
        filename: Output file name
        """
        np.savez(f"Models/{filename}", weights_out=self.weights_out,
                 weights_hidden=self.weights_hidden,biases_out=self.biases_out,
                 biases_hidden=self.biases_hidden)

    def load_model(self, filename):
        """
        Loads a saved model containing the weights and biases.
        filename: File name
        """
        model = np.load(f"Models/{filename}", allow_pickle=True)
        self.weights_out = model["weights_out"]
        self.weights_hidden = model["weights_hidden"]
        self.biases_out = model["biases_out"]
        self.biases_hidden = model["biases_hidden"]
        self.n_features = self.weights_hidden[0].shape[0]
        self.n_hidden_layers = len(self.weights_hidden)

    @staticmethod
    @njit
    def activation_function_out(z_i, activation_function_output):
        """
        Method for modifiable output activation function.
        Supports linear and sigmoid.
        z_i: ndarray - Activation of node
        activation_function_output: Either "linear" for a linear output
                activation function or "sigmoid" for the sigmoid as output
                activation function
        """
        if activation_function_output == "linear":
            return z_i
        elif activation_function_output == "sigmoid":
            expo = np.exp(z_i)
            return expo/(1 + expo)

    @staticmethod
    @njit
    def grad_activation_out(z_i, activation_function_output):
        """
        Returns the gradient of the output activation function.
        Supports linear and sigmoid.
        z_i: ndarray - Activation of node
        activation_function_output: Either "linear" for a linear output
                activation function or "sigmoid" for the sigmoid as output
                activation function
        """
        if activation_function_output == "linear":
            return np.ones_like(z_i)
        elif activation_function_output == "sigmoid":
            expo = np.exp(-z_i)
            return expo/((1 + expo)**2)

    @staticmethod
    @njit
    def sigmoid(z):
        """
        Returns the sigmoid function. Use as activation function
        z: ndarray - Activation of node
        """
        expo = np.exp(z)
        return expo/(1 + expo)

    @staticmethod
    @njit
    def grad_activation(z_i):
        """
        Returns the gradient of the sigmoid function.
        z: ndarray - Activation of node
        """
        expo = np.exp(-z_i)
        return expo/((1 + expo)**2)

    def cost_func(self, y, y_pred):
        """
        Returns the cost function values with added penalty term
        y: ndarray - The true binary outputs
        y_pred: ndarray - The predicted binary outputs found using the model
        """
        C = (-np.sum(sps.xlogy(y, y_pred) + sps.xlogy(1 - y, 1 - y_pred))
            /y_pred.shape[0]) + self.lambd*self.l2
        return C

    @staticmethod
    @njit
    def grad_cost(y, y_pred):
        """
        Returns the gradient of the cross-entropy/cost function values.
        y: ndarray - The true binary outputs
        y_pred: ndarray - The predicted binary outputs found using the model
        """
        return y_pred - y

    @property
    def l2(self):
        """
        Calculates L2 penalty on the weights
        """
        sum_weights = 0
        for weights in self.weights_hidden:
            sum_weights += (weights**2).sum()
        return (sum_weights + (self.weights_out**2).sum())/(2*self.n_features)


class MLPRegressor(MLPClassifier):
    def __init__(self, hidden_layer_size=[20, 10], learning_rate=0.1, lambd=0,
                 n_epochs=2000, rtol=0.001, batch_size="auto", verbose=False,
                 activation_function_output="linear", learning_schedule=None):
        """
        Performs regression analysis using a multilayer perceptron
        hidden_layer_size: iterable, default [20, 10] - Number of nodes in
                each hidden layer.
        lambd: float, default 0 - The L2 shrinkage parameter used for
                regularisation of the weights.
        learning rate: float, default 0.1 - The learning rate used in the
                stochastic gradient descent solver.
        n_epochs: int, default 2000 - Number of epochs used in the stochastic
                gradient descent solver.
        rtol: float, default 0.001 - Relative tolerance used as a stopping
                criteria in the stochastic gradient descent solver.
        batch_size: int, default "auto" - Size of the minibatches used in the
                stochastic gradient descent solver.
        verbose: bool, default False - If True, print progress in terminal.
        activation_function_output: str, default "linear" -
            Specifies which activation function to use for the output layer.
            Supports "linear" and "sigmoid".
        learning_schedule: float, default None - Deprecated.
        """
        super().__init__(hidden_layer_size, learning_rate, lambd, n_epochs, rtol,
             batch_size, verbose, activation_function_output, learning_schedule)

    def predict(self, X):
        """
        Converts output probabilities into binary output
        X: ndarray - Design matrix
        """
        if self.weights_hidden[0].shape[0] != X.shape[1]:
            print(len(self.weights_hidden[0].shape[0]), X.shape[1])
            raise ValueError("Model was fitted on different inputs than what was provided")
        pred = self.feed_forward(X)[0][-1]
        return pred

    def cost(self, y, y_pred):
        """
        Returns the cost function, (MSE)/2 with added penalty term
        y: ndarray - The true binary outputs
        y_pred: ndarray - The predicted binary outputs found using the model
        """
        C = np.mean((y_pred - y)**2)/2 + self.lambd*self.l2
        return C

    @staticmethod
    @njit
    def grad_cost(y, y_pred):
        """
        Returns the gradient of the cost function, (MSE)/2.
        y: ndarray - The true binary outputs
        y_pred: ndarray - The predicted binary outputs found using the model
        """
        return y_pred - y

    def accuracy_score(self):
        """
        Raises an error if used here.
        """
        raise TypeError("Accuracy score is not valid for regression")

    def r2_score(self, X, y):
        """
        Returns R2-score of the model
        X: ndarray - Design matrix
        y: ndarray - True binary outputs
        """
        R2 = 1 - np.sum((y - self.predict(X))**2)/np.sum((y - np.mean(y))**2)
        return R2


class log_uniform():
    """
    Uses the scikit-learn sklearn.model_selection.RandomizedSearchCV function
    to make a logistic uniform distribution to find optimal parameters for
    the subclasses LogReg, MLPClassifier and MLPRegression.
    """
    def __init__(self, a, b):
        self.loc = a
        self.scale = b - a

    def rvs(self, size=None, random_state=None):
        uniform = scipy.stats.uniform(self.loc, self.scale)
        if size == None:
            return 10**uniform.rvs(random_state=random_state)
        else:
            return 10**uniform.rvs(size=size, random_state=random_state)

if __name__ == "__main__":
    pass