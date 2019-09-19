import numpy as np
import sklearn.linear_model as skl
import sklearn.metrics as sklm
from sklearn.model_selection import train_test_split, KFold
from plots import plot3d
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

def MSE(y_data, y_model):
    """Mean Squared Error function"""
    y_data = np.ravel(y_data)
    y_model = np.ravel(y_model)
    return np.mean((y_data-y_model)**2)

def R2score(y_data, y_model):
    """R^2 score function"""
    y_data = np.ravel(y_data)
    y_model = np.ravel(y_model)
    return 1 - np.sum((y_data-y_model)**2)/np.sum((y_data-np.mean(y_data))**2)

def RelativeError(y_data, y_model):
    """Relative error"""
    y_data = np.ravel(y_data)
    y_model = np.ravel(y_model)
    return abs((y_data-y_model)/y_data)

def Var(ytilde):
    """Variance"""
    y_tilde = np.ravel(ytilde)
    return np.mean((y_tilde - np.mean(y_tilde))**2)

def Bias(ytilde, y):
    """Bias"""
    y = np.ravel(y)
    y_tilde = np.ravel(ytilde)
    return np.mean((y - np.mean(y_tilde))**2)

def TrainData(M, a, test=0.25):
    """Split data in training data and test data"""
    z = np.ravel(a)
    X_train, X_test, Z_train, z_test = train_test_split(M, z, test_size=test)
    return X_train, X_test, Z_train, z_test

def OLS(X, y):
    """Ordinary least squared using singular value decomposition (SVD)"""
    U, s, VT = np.linalg.svd(X)
    D = np.diag(s**2)
    Xinv = np.linalg.inv(VT.T @ D @ VT)
    beta = Xinv @ X.T @ y
    return beta

def Ridge(X, y, lamb):
    """Ridge regression"""
    beta_OLS = OLS(X, y)
    beta_ridge = beta_OLS*1./(1.+lamb)
    return beta_ridge
    
def Lasso(X, z, alpha):
    """Lasso regression"""
    clf = skl.Lasso(alpha).fit(X, np.ravel(z))
    beta = clf.coef_
    return beta

def confidence_int(X, z, z_tilde, beta):
    """Confidence interval of beta"""
    varbeta = np.sqrt(np.linalg.inv(X.T @ X)).diagonal()
    percent = [99, 98, 95, 90]
    Z = [2.576, 2.326, 1.96, 1.645]
    sigmaSQ = np.sum((z - z_tilde)**2)/(len(z) - len(beta) - 1)
    for k in range(len(beta)):
        print("Confidence interval for beta %i" % (k + 1))
        for i, n in enumerate(percent):
            print("%2i%%: %3.2f +- %3.2f" % (percent[i], beta[k], Z[i]*np.sqrt(sigmaSQ)*varbeta[k]))

def k_fold_CV(X, y, folds):
    """k-fold cross-validation"""
    X_train, X_test, Z_train, Z_test = TrainData(X, y, test=0.25)
    x_train = np.split(X_train, folds)
    z_train = np.split(Z_train, folds)
    x_test = np.split(X_test, folds)
    z_test = np.split(Z_test, folds) 
    
    MSE_train = []
    MSE_test =[]
    R2_train = []
    R2_test = []
    for i in range(folds):
        X_train = x_train
        X_train = np.delete(X_train, i, 0)
        X_train = np.concatenate(X_train)
        X_test = x_test
        X_test = np.delete(X_test, i, 0)
        X_test = np.concatenate(X_test)

        Z_train = z_train
        Z_train = np.delete(Z_train, i, 0)
        Z_train = np.ravel(Z_train)
        Z_test = z_test
        Z_test = np.delete(Z_test, i, 0)
        Z_test = np.ravel(Z_test)

        beta_train = OLS(X_train, Z_train)
        y_tilde_train = X_train @ beta_train
        beta_test = OLS(X_test, Z_test)
        y_tilde_test = X_test @ beta_test

        MSE_train_i = MSE(Z_train, y_tilde_train)
        R2_train_i = R2score(Z_train, y_tilde_train)
        MSE_test_i = MSE(Z_test, y_tilde_test)
        R2_test_i = R2score(Z_test, y_tilde_test)

        MSE_train = np.append(MSE_train, MSE_train_i)
        R2_train = np.append(R2_train, R2_train_i)
        MSE_test = np.append(MSE_test, MSE_test_i)
        R2_test = np.append(R2_test, R2_test_i)

    return np.mean(MSE_train), np.mean(R2_train), np.mean(MSE_test), np.mean(R2_test)

N = 20
n = 5

x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))

X, Y = np.meshgrid(x, y)
noise = 0.8*np.random.normal(0, 1, size=X.shape)
Z = FrankeFunction(X, Y) + noise
z = np.ravel(Z)

M = DesignMatrix(X, Y, n)
#print(M)
#print(M.size, M.shape)
#print(Z.size, Z.shape)
#print(X.size, X.shape)
"""
beta_OLS = OLS(X_train, Z_train)
y_tilde = X_train @ beta_OLS
#print(X_train.size, X_train.shape)
#print(beta_OLS.size, beta_OLS.shape)
#print(y_tilde.size, y_tilde.shape)
#print(Z.size, Z.shape)
#print(z.size, z.shape)
beta = OLS(X_train, Z_train)
y_tilde = M @ beta
#print(beta)
#mse = MSE(Z_test, y_tilde)
#r2score = R2score(Z_test, y_tilde)
#print(mse)
#print(r2score)
"""

def fig_bias_var(x, y, p=10, n=20):
    error_MSE = np.zeros((4, p+1))
    error_R2 = np.zeros((4, p+1))
    
    error_MSE_train = np.zeros((4, p+1))
    error_R2_train = np.zeros((4, p+1))
    
    complexity = np.arange(0, p+1, 1)
    
    for i in range(n):
        Z = FrankeFunction(x, y) #+ np.random.normal(0, 1, size=x.shape)
        print(i)
        for j in range(p+1):
            X_train, X_test, Z_train, Z_test = TrainData(M, Z, test=0.25)
            # Test data
            beta_OLS = OLS(X_train, Z_train)
            beta_Ridge = Ridge(X_train, Z_train, lamb=0.1)
            beta_k, _, _ = k_fold_CV(X_train, Z_train, 5, shuffle = False)
            beta_Lasso = Lasso(X_train, Z_train, alpha=0.000001)
            
            z_tilde_OLS = X_test @ beta_OLS
            z_tilde_Ridge = X_test @ beta_Ridge
            z_tilde_k = X_test @ beta_k
            z_tilde_Lasso = X_test @ beta_Lasso
            
            error_MSE[0, j] += MSE(Z_test, z_tilde_OLS)
            error_MSE[1, j] += MSE(Z_test, z_tilde_k)
            error_MSE[2, j] += MSE(Z_test, z_tilde_Ridge)
            error_MSE[3, j] += MSE(Z_test, z_tilde_Lasso)
            error_R2[0, j] += R2score(Z_test, z_tilde_OLS)
            error_R2[1, j] += R2score(Z_test, z_tilde_k)
            error_R2[2, j] += R2score(Z_test, z_tilde_Ridge)
            error_R2[3, j] += R2score(Z_test, z_tilde_Lasso)
            # Training data
            z_tilde_OLS = X_train @ beta_OLS
            z_tilde_k = X_train @ beta_k
            z_tilde_Ridge = X_train @ beta_Ridge
            z_tilde_Lasso = X_train @ beta_Lasso
            
            error_MSE_train[0, j] += MSE(Z_train, z_tilde_OLS)
            error_MSE_train[1, j] += MSE(Z_train, z_tilde_k)
            error_MSE_train[2, j] += MSE(Z_train, z_tilde_Ridge)
            error_MSE_train[3, j] += MSE(Z_train, z_tilde_Lasso)
            error_R2_train[0, j] += R2score(Z_train, z_tilde_OLS)
            error_R2_train[1, j] += R2score(Z_train, z_tilde_k)
            error_R2_train[2, j] += R2score(Z_train, z_tilde_Ridge)
            error_R2_train[3, j] += R2score(Z_train, z_tilde_Lasso)

    error_MSE /= n
    error_R2 /= n
    error_MSE_train /= n
    error_R2_train /= n

    plt.title('OLS')
    plt.plot(complexity, error_MSE[0], label = 'Test')
    plt.plot(complexity, error_MSE_train[0], label = 'Training')
    plt.ylim([0, np.max(error_MSE[0]*1.2)])
    plt.legend()

    plt.title('k-fold')
    plt.plot(complexity, error_MSE[1], label = 'Test')
    plt.plot(complexity, error_MSE_train[1], label = 'Training')
    plt.ylim([0, np.max(error_MSE[1]*1.2)])

    plt.title('Ridge')
    plt.plot(complexity, error_MSE[2], label = 'Test')
    plt.plot(complexity, error_MSE_train[2], label = 'Training')
    plt.ylim([0, np.max(error_MSE[2]*1.2)])
    plt.legend()

    plt.title('Lasso')
    plt.plot(complexity, error_MSE[3], label = 'Test')
    plt.plot(complexity, error_MSE_train[3], label = 'Training')
    plt.ylim([0, np.max(error_MSE[3]*1.2)])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    """OLS on Franke function"""
    print("Before train_test:")
    Z = FrankeFunction(X, Y)
    z = np.ravel(Z)
    beta_OLS = OLS(M, z)
    y_tilde = M @ beta_OLS
    mse = MSE(Z, y_tilde)
    r2score = R2score(Z, y_tilde)
    var = Var(y_tilde)
    print("MSE =", mse)
    print("R2-score =", r2score)
    print("Variance =", var)
    #conf_int = confidence_int(M, z, y_tilde, beta_OLS)
    #plot3d(X, Y, z=np.reshape(y_tilde, Z.shape), z2=Z)
    """Resampling"""
    print("Resampling of test data with k_fold:")
    Z = FrankeFunction(X, Y) + noise
    MSE_train, R2_train, MSE_test, R2_test = k_fold_CV(M, Z, folds=5)
    print("MSE train set =", MSE_train)
    print("R2-score train set =", R2_train)
    print("MSE test set =", MSE_test)
    print("R2-score test set =", R2_test)
    """Bias-variance tradeoff"""
    #fig_bias_var(X, Y, p=10, n=20)
    
"""
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.figure()
plt.title("a)")
plt.scatter(x, y)
plt.plot(x, ytilde, color="red")
plt.xlabel("X")
plt.ylabel("Y")
fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
lineg = skl.LinearRegression().fit(M,y_vec)
ypredict = lineg.predict(M)
plt.figure()
plt.title("b)")
plt.scatter(x_vec, y_vec, color="black")
#plt.plot(x, ytilde, color="blue", label="Custom", marker="o")
plt.plot(x_vec, ypredict, color="red", label="Sklearn")
plt.xlabel("X")
plt.ylabel("Y")
test_mse = sklm.mean_squared_error(Z_train, ytilde)
test_r2 = sklm.r2_score(Z_train, ytilde)
print(test_mse)
print(test_r2)
"""

def sigma_sqr(X, z, ztilde):
    """Sigma squared (variance?)"""
    z = np.ravel(z)
    z_tilde = np.ravel(ztilde)
    return 1./(len(z) - len(X[0]) - 1)*np.sum((z_tilde - z)**2)
    
def beta_var(X, z, ztilde):
    """Beta-variance"""
    U, s, VT = np.linalg.svd(X)
    D = np.diag(s**2)
    sigma = np.zeros(X.shape)
    sigma[:len(s), :len(s)] = np.linalg.inv(D)
    var = sigma_sqr(X, z, ztilde)*VT @ sigma.T @ sigma @ VT.T
    return np.linalg.inv(var)
