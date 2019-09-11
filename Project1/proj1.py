import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split

#a)
def DesignMatrix(x, y, n):
    """Create design matrix"""
    N = len(x)
    num = int((n+1)*(n+2)/2.)
    M = np.ones((N, num))
    
    for i in range(1, n+1):
        q = int(i*(i+1)/2.)
        for j in range(i+1):
            M[:,q+j] = x**(i-j)*y**j
            
    return M

def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def MSE(y_data, y_model):
    """Mean Squared Error function"""
    return np.mean((y_data-y_model)**2)

def R2score(y_data, y_model):
    """R^2 score function"""
    return 1 - np.sum((y_data-y_model)**2)/np.sum((y_data-np.mean(y_data))**2)

def RelativeError(y_data, y_model):
    """Relative error"""
    return abs((y_data-y_model)/y_data)

def TrainData(M, a, test=0.25):
    """Split data in training data and test data"""
    X_train, X_test, Z_train, z_test = train_test_split(M, z, test_size=test)
    return X_train, X_test, Z_train, z_test

def OLS(X, y):
    """Ordinary least squared"""
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    #print(beta)
    y_tilde = X @ beta
    return beta, y_tilde

N = 10
n = 5

x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))

X, Y = np.meshgrid(x, y)
Z = FrankeFunction(X, Y)

x_vec = np.ravel(X)
y_vec = np.ravel(Y)
m = int(len(x_vec))
noise = np.random.random(m)

z = np.ravel(Z) + noise

M = DesignMatrix(x_vec, y_vec, n)

X_train, X_test, Z_train, Z_test = TrainData(M, z, test=0.25)

beta, ytilde = OLS(X_train, Z_train)

mse = MSE(Z_train, ytilde)
r2score = R2score(Z_train, ytilde)
print(mse)
print(r2score)
