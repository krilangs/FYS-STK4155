import numpy as np
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import scipy

np.random.seed(1337)

def DesignMatrix(x, y, n):
    """Create design matrix"""
    x = np.ravel(x)
    y = np.ravel(y)
    N = len(x)
    num = int((n+1)*(n+2)/2.)
    M = np.ones((N, num))
    
    for i in range(1, n+1):
        q = int(i*(i+1)/2.)
        for j in range(i+1):
            M[:, q+j] = x**(i-j)*y**j    
    return M

def FrankeFunction(x, y):
    """Franke function"""
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def MSE(data, model):
    """Mean Squared Error function"""
    y_data = np.ravel(data)
    y_model = np.ravel(model)
    return np.mean((y_data-y_model)**2)

def R2score(data, model):
    """R^2 score function"""
    y_data = np.ravel(data)
    y_model = np.ravel(model)
    return 1 - np.sum((y_data-y_model)**2)/np.sum((y_data-np.mean(y_data))**2)

def VAR(model):
    """Variance"""
    y_model = np.ravel(model)
    return np.mean((y_model - np.mean(y_model))**2)

def BIAS(data, model):
    """Bias"""
    y_model = np.ravel(model)
    y_data = np.ravel(data)
    return np.mean((y_data - np.mean(y_model))**2)

def TrainData(M, v, test=0.25):
    """Split data in training data and test data"""
    z = np.ravel(v)
    X_train, X_test, Z_train, Z_test = train_test_split(M, z, test_size=test, shuffle=True)
    return X_train, X_test, Z_train, Z_test

def OLS(X, data):
    """Ordinary least squared using singular value decomposition (SVD)"""
    U, s, VT = np.linalg.svd(X)
    D = np.diag(s**2)
    Xinv = np.linalg.inv(VT.T @ D @ VT)
    beta = Xinv @ X.T @ data
    return beta

def Ridge(X, data, hyperparam):
    """Ridge regression"""
    U, s, V = np.linalg.svd(X)
    sigma = np.zeros(X.shape)
    sigma[:len(s), :len(s)] = np.diag(s)
    inverse = scipy.linalg.inv(sigma.T.dot(sigma) + hyperparam*np.identity(len(s)))
    beta = V.T.dot(inverse).dot(sigma.T).dot(U.T).dot(data)
    return beta
    
def Lasso(X, data, hyperparam):
    """Lasso regression"""
    clf = skl.Lasso(alpha=hyperparam, max_iter=1e5, tol=0.1).fit(X, data)
    beta = clf.coef_
    beta[0] = clf.intercept_
    return beta

def confidence_int(x, y, z, hyperparam, method=""):
    """Confidence interval of beta"""
    X = DesignMatrix(x, y, n=5)
    if method == "OLS":
        beta = OLS(X, np.ravel(z))
    elif method == "Ridge":
        beta = Ridge(X, np.ravel(z), hyperparam)
    elif method == "Lasso":
        beta = Lasso(X, z, hyperparam)
    ztilde = X @ beta
    E, P = np.linalg.eigh(X.T @ X)
    D_inv = np.diag(1/E)
    varbeta = np.sqrt(P @ D_inv @ P.T).diagonal()
    zSTD = np.sum((z - ztilde)**2)/(len(z) - len(beta) - 1)
    betaSTD = np.sqrt(zSTD)*varbeta
    Z = [2.576, 2.326, 1.96, 1.645]
    """
    percent = [99, 98, 95, 90]
    for k in range(len(beta)):
        print("Confidence interval for beta %i" % (k + 1))
        for i, n in enumerate(percent):
            print("%2i%%: %3.2f +- %3.2f" % (percent[i], beta[k], Z[i]*betaSTD[k]))
    """
    return Z[1]*betaSTD

def k_fold_CV(x, y, z, folds, dim, hyperparam, method="", train=False):
    """k-fold cross-validation"""
    Mse = np.zeros(folds)
    R2 = np.zeros(folds)
    Var = np.zeros(folds)
    Bias = np.zeros(folds)
    if train:
        Mse_train = np.zeros(folds)
    
    X_shuffle, Y_shuffle, Z_shuffle = shuffle(x, y, z)
    
    x_split = np.array_split(X_shuffle, folds)
    y_split = np.array_split(Y_shuffle, folds)
    z_split = np.array_split(Z_shuffle, folds)
    
    for i in range(folds):
        X_test = x_split[i]
        Y_test = y_split[i]
        Z_test = z_split[i]
        
        X_train = np.delete(x_split, i, axis=0).ravel()
        Y_train = np.delete(y_split, i, axis=0).ravel()
        Z_train = np.delete(z_split, i, axis=0).ravel()
        
        X_train = DesignMatrix(X_train, Y_train, dim)
        X_test = DesignMatrix(X_test, Y_test, dim)
        if method == "OLS":
            beta = OLS(X_train, Z_train)
        elif method == "Ridge":
            beta = Ridge(X_train, Z_train, hyperparam)
        elif method == "Lasso":
            beta = Lasso(X_train, Z_train, hyperparam)
        
        
        z_fit = X_test @ beta
        Mse[i] = MSE(Z_test, z_fit)
        if train:
            z_train = X_train @ beta
            Mse_train[i] = MSE(Z_train, z_train)
        R2[i] = R2score(Z_test, z_fit)
        Var[i] = VAR(z_fit)
        Bias[i] = BIAS(Z_test, z_fit)
    if train:
        return np.mean(Mse), np.mean(Mse_train)
    else:
        return np.mean(Mse), np.mean(R2), np.mean(Var), np.mean(Bias)
