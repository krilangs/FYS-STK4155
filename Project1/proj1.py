from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
import sklearn.metrics as sklm
from sklearn.model_selection import train_test_split
np.random.seed(1337)

def DesignMatrix(x, y, n):
    """Create design matrix"""
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
    
def Lasso(alpha, X, z):
    """Lasso regression"""
    clf = skl.Lasso(alpha).fit(X, np.ravel(z))
    beta = clf.coef_
    return beta

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

def confidence_int(X, z_tilde, beta):
    """Confidence interval of beta"""
    varbeta = np.sqrt(np.linalg.inv(X.T @ X)).diagonal()
    percent = [99, 98, 95, 90]
    Z = [2.576, 2.326, 1.96, 1.645]
    sigmaSQ = np.sum((z - z_tilde)**2)/(len(z) - len(beta) - 1)
    for k in range(len(beta)):
        print("Confidence interval for beta %i" % (k + 1))
        for i, n in enumerate(percent):
            print("%2i%%: %3.2f +- %3.2f" % (percent[i], beta[k], Z[i]*np.sqrt(sigmaSQ)*varbeta[k]))

def k_fold_CV(X, y, folds, shuffle = False):
    """k-fold cross-validation"""
    if shuffle == True:
        interval = np.random.choice(len(y), replace = False, size =int(len(y)))
        isplit = np.sort(np.array_split(interval, folds))
    else:
        interval = np.arange(len(y))
        isplit = np.array_split(interval, folds)
    kR2 = 0
    kMSE = 0
    for i in range(folds):
        X_train, y_train = np.ma.array(X, mask = False), np.ma.array(y, mask = False)
        y_train.mask[isplit[i]] = True
        X_train.mask[isplit[i],:] = True
        X_train = np.ma.compress_rows(X_train)
        y_train = y_train.compressed()
        beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        y_tilde = X_train @ beta
        kR2 += R2score(y_train, y_tilde)
        kMSE += MSE(y_train, y_tilde)
    return kR2/folds, kMSE/folds


N = 10
n = 5

x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))

X, Y = np.meshgrid(x, y)
noise = np.random.normal(0, 1, size=X.shape)
Z = FrankeFunction(X, Y) + noise
z = np.ravel(Z)

x_vec = np.ravel(X)
y_vec = np.ravel(Y)
#m = int(len(x_vec))
M = DesignMatrix(x_vec, y_vec, n)

X_train, X_test, Z_train, Z_test = TrainData(M, z, test=0.25)
beta_OLS = OLS(X_train, Z_train)
y_tilde = X_train @ beta_OLS

mse = MSE(Z_train, y_tilde)
r2score = R2score(Z_train, y_tilde)
print(mse)
print(r2score)








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
