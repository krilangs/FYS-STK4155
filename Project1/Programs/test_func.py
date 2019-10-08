import numpy as np
import sklearn.model_selection as sklms
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from funcs import FrankeFunction, DesignMatrix, OLS, k_fold_CV_franke, Ridge

np.random.seed(777)

def test_OLS(N, n):
    print("Test OLS against scikit:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    M = DesignMatrix(X, Y, n)

    Z = FrankeFunction(X, Y)
    z = np.ravel(Z)
    X_train, X_test, Z_train, Z_test = train_test_split(M, z, test_size=0.25, shuffle=True)
    beta_OLS = OLS(M, z)
    ytilde = X_test @ beta_OLS
    # Scikit
    OLS_scikit = skl.LinearRegression().fit(X_train, Z_train)
    evalmodel_scikit = OLS_scikit.predict(X_test)
    np.testing.assert_allclose(evalmodel_scikit, ytilde, rtol=0.6)

def test_Ridge(N, n, hyperparam):
    print("Test Ridge against scikit:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    M = DesignMatrix(X, Y, n)

    Z = FrankeFunction(X, Y)
    z = np.ravel(Z)
    X_train, X_test, Z_train, Z_test = train_test_split(M, z, test_size=0.25, shuffle=True)
    beta_Ridge = Ridge(M, z, hyperparam)
    ytilde = X_test @ beta_Ridge
    # Scikit
    ridge_scikit = skl.Ridge(alpha=hyperparam).fit(X_train, Z_train)
    evalmodel_scikit = ridge_scikit.predict(X_test)
    np.testing.assert_allclose(evalmodel_scikit, ytilde, rtol=0.3)

def test_KFold(N, n):
    print("Test k-fold with scikit:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    M = DesignMatrix(X, Y, n)

    noise = 0.2*np.random.normal(0, 1, size=X.shape)
    Z = FrankeFunction(X, Y) + noise
    z = np.ravel(Z)

    Mse, R2, Var, Bias = k_fold_CV_franke(X, Y, Z, folds=5, dim=5, hyperparam=1,
                                          method="OLS", train=False)
    print("MSE k-fold = ", Mse)
    # Scikit
    X_train, X_test, Z_train, Z_test = train_test_split(M, z, test_size=0.25, shuffle=True)
    kfold = sklms.KFold(n_splits=5, shuffle=True)
    OLS_scikit = skl.LinearRegression()
    MSE_scikit = np.mean(-sklms.cross_val_score(OLS_scikit, X_train, Z_train,
                        scoring="neg_mean_squared_error", cv=kfold, n_jobs=-1))
    np.testing.assert_allclose(Mse, MSE_scikit,rtol=0.05)
    print("MSE scikit = ", MSE_scikit)

#test_OLS(N=30, n=5)
#test_Ridge(N=40, n=5, hyperparam=1e-5)
#test_KFold(N=50, n=5)
