import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
import sklearn.metrics as sklm
from sklearn.model_selection import train_test_split

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
    

def k_fold_CV(folds):
    """k-fold cross-validation"""
    return ...


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
beta, ytilde = OLS(X_train, Z_train)

mse = MSE(Z_train, ytilde)
r2score = R2score(Z_train, ytilde)
print(mse)
print(r2score)

"""
plt.figure()
plt.title("a)")
plt.scatter(x, y)
plt.plot(x, ytilde, color="red")
plt.xlabel("X")
plt.ylabel("Y")
"""
lineg = skl.LinearRegression().fit(M,y_vec)
ypredict = lineg.predict(M)

plt.figure()
plt.title("b)")
plt.scatter(x_vec, y_vec, color="black")
#plt.plot(x, ytilde, color="blue", label="Custom", marker="o")
plt.plot(x_vec, ypredict, color="red", label="Sklearn")
plt.xlabel("X")
plt.ylabel("Y")


"""
test_mse = sklm.mean_squared_error(Z_train, ytilde)
test_r2 = sklm.r2_score(Z_train, ytilde)
print(test_mse)
print(test_r2)
"""
