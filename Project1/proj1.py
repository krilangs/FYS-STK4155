import numpy as np
import sklearn.linear_model as skl
import sklearn.metrics as sklm
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from plots import plot3d, plot_conf_int, fig_bias_var
import matplotlib.pyplot as plt
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

#def RelativeError(data, model):
#    """Relative error"""
#    y_data = np.ravel(data)
#    y_model = np.ravel(model)
#    return abs((y_data-y_model)/y_data)

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
    clf = skl.Lasso(alpha=hyperparam, max_iter=5e5, tol=0.015).fit(X, np.ravel(data))
    beta = clf.coef_
    beta[0] = clf.intercept_
    return beta

def confidence_int(x, y, z, hyperparam, method=""):
    """Confidence interval of beta"""
    X = DesignMatrix(x, y, n=5)
    if method == "OLS":
        beta = OLS(X, np.ravel(z))
    if method == "Ridge":
        beta = Ridge(X, np.ravel(z), hyperparam)
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

    
if __name__ == "__main__":
    N = 40
    n = 5

    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    noise = 0.8*np.random.normal(0, 1, size=X.shape)
    Z = FrankeFunction(X, Y) + noise
    z = np.ravel(Z)

    M = DesignMatrix(X, Y, n)
    #-----------------------------------
    """OLS on Franke function"""
    print("a) Before train_test without noise:")
    Z = FrankeFunction(X, Y)
    z = np.ravel(Z)
    beta_OLS = OLS(M, z)
    y_tilde = M @ beta_OLS
    mse_OLS = MSE(Z, y_tilde)
    r2score_OLS = R2score(Z, y_tilde)
    var_OLS = VAR(y_tilde)

    print("MSE =", mse_OLS)
    print("R2-score =", r2score_OLS)
    print("Variance =", var_OLS)
    N = 100
    #plot_conf_int(N, hyperparam=1, method="OLS")
    #plot3d(X, Y, Z, Z+noise)
    #-----------------------------------
    """Resampling"""
    print("b) Resampling of test data with k_fold:")
    Z = FrankeFunction(X, Y) + noise
    #Mse, R2, Var, Bias = k_fold_CV(X, Y, Z, folds=5, dim=5, hyperparam=1, method="OLS", train=False)

    #print("MSE k-fold =", Mse)
    #print("R2-score k-fold =", R2)
    #print("Variance k-fold=", Var)
    #print("Bias k-fold =", Bias)
    #------------------------------------
    """Bias-variance tradeoff"""
    print("c) Plotting bias-variance tradeoff:")
    #fig_bias_var(X, Y, hyperparam=1, p=12, method="OLS")
    #------------------------------------
    """Ridge Regression"""
    print("d) Ridge analysis:")
    """
    lambda_Ridge = np.logspace(-7, 1, 9)
    #print(lambda_Ridge)
    Z = FrankeFunction(X, Y)
    z = np.ravel(Z)
    for i in lambda_Ridge: 
        beta_Ridge = Ridge(M, z, hyperparam=i)
        y_tilde = M @ beta_Ridge
        mse_Ridge = MSE(Z, y_tilde)
        r2score_Ridge = R2score(Z, y_tilde)
        var_Ridge = VAR(y_tilde)
        #print("Lambda=",i)
        #print("MSE =", mse_Ridge)
        #print("R2-score =", r2score_Ridge)
        #print("Variance =", var_Ridge)
    lambda_params = [10, 0.1, 1e-3, 1e-6, 1e-10]
    Z = FrankeFunction(X, Y) + noise
    #for j in lambda_params:
    #    plot_conf_int(N, hyperparam=j, method="Ridge")
    
        #Mse, R2, Var, Bias = k_fold_CV(X, Y, Z, folds=5, dim=5, hyperparam=1, method="Ridge", train=False)
        #print("MSE k-fold =", Mse)
        #print("R2-score k-fold =", R2)
        #print("Variance k-fold=", Var)
        #print("Bias k-fold =", Bias)

        #fig_bias_var(X, Y, hyperparam=j, p=12, method="Ridge")
    """
    #------------------------------------
    """Lasso Regression"""
    print("e) Lasso analysis")
    Z = FrankeFunction(X, Y)
    z = np.ravel(Z)
    beta_Lasso = Lasso(M, z, hyperparam=0.01)  # lambda=0 should give ~OLS
    y_tilde = M @ beta_Lasso
    mse_Lasso = MSE(Z, y_tilde)
    r2score_Lasso = R2score(Z, y_tilde)
    var_Lasso = VAR(y_tilde)

    print("MSE =", mse_Lasso)
    print("R2-score =", r2score_Lasso)
    print("Variance =", var_Lasso)






