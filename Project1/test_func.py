import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.linear_model as skl
from proj1 import FrankeFunction, DesignMatrix, OLS, TrainData, k_fold_CV

np.random.seed(1337)

def test_OLS(N, n):
    print("Test OLS against scikit:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    M = DesignMatrix(X, Y, n)

    Z = FrankeFunction(X, Y)
    z = np.ravel(Z)
    X_train, X_test, Z_train, Z_test = TrainData(M, Z, test=0.25)
    beta_OLS = OLS(M, z)
    OLS_scikit = skl.LinearRegression().fit(X_train, Z_train)
    evalmodel_scikit = OLS_scikit.predict(X_test)
    print(len(evalmodel_scikit))
    ytilde = X_test @ beta_OLS
    print(len(ytilde))
    np.testing.assert_allclose(evalmodel_scikit, ytilde, rtol=0.05)
    
def test_KFold(N, n):
    print("Test k-fold with scikit:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    M = DesignMatrix(X, Y, n)
    
    noise = 0.8*np.random.normal(0, 1, size=X.shape)
    Z = FrankeFunction(X, Y) + noise

    Mse, R2, Var, Bias = k_fold_CV(X, Y, Z, folds=5, dim=5, hyperparam=1, reg="OLS", train=False)
    print("MSE k-fold = ", Mse)
    X_train, X_test, Z_train, Z_test = TrainData(M, Z, test=0.25)
    kfold = sklms.KFold(n_splits=5, shuffle=True)
    OLS_scikit = skl.LinearRegression()
    EPE_scikit = np.mean(-sklms.cross_val_score(OLS_scikit, X_train, Z_train,
                        scoring="neg_mean_squared_error", cv=kfold, n_jobs=-1))
    np.testing.assert_allclose(Mse, EPE_scikit,rtol=0.05)
    print("MSE scikit = ", EPE_scikit)
    
test_KFold(N=30, n=5)
test_OLS(N=100, n=5)