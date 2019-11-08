import numpy as np

from sklearn.preprocessing import  StandardScaler
from main import MLPClassifier, MLPRegressor

np.random.seed(777)

def MLPClassifier_test():
    """
    Test the MLPClassifier class for overfitting
    """
    print("Test MLP classification")
    y = np.zeros(1000)
    X = np.random.randint(0, 2, size=y.shape)
    y[X == 1] = 1
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = MLPClassifier(n_epochs=50, batch_size=32, hidden_layer_size=[10],
                          learning_rate=1, verbose=False)
    model.fit(X, y)
    y_model = model.predict(X)
    np.testing.assert_array_equal(y, y_model)

def MLPRegressor_test():
    """
    Test the MLPRegressor class for overfitting
    """
    print("Test MLP regression")
    x = np.linspace(0, 5, 10000, endpoint=True)
    y = 3 * x
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    scalerx = StandardScaler().fit(x)
    x = scalerx.transform(x)
    model = MLPRegressor(n_epochs=100, hidden_layer_size=[3], rtol=-np.inf,
                         learning_rate=2e-3, verbose=False)
    model.fit(x, y)
    y_model = model.predict(x)
    np.testing.assert_array_almost_equal(y, y_model, decimal=1)
