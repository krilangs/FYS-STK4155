import numpy as np
from sklearn.model_selection import train_test_split
from funcs import *
from plots import plot3d, plot_conf_int, fig_bias_var

np.random.seed(1337)

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
#plot_conf_int(N=100, hyperparam=1, method="OLS")
#plot3d(X, Y, Z, Z+noise)

#---------------------------------------------------------
"""Resampling"""
print("b) Resampling of test data with k_fold:")
Z = FrankeFunction(X, Y) + noise
Mse, R2, Var, Bias = k_fold_CV(X, Y, Z, folds=5, dim=5, hyperparam=1, method="OLS", train=False)

print("MSE k-fold =", Mse)
print("R2-score k-fold =", R2)
print("Variance k-fold=", Var)
print("Bias k-fold =", Bias)
#---------------------------------------------------------

"""Bias-variance tradeoff"""
print("c) Plotting bias-variance tradeoff:")
#fig_bias_var(X, Y, p=12, method="OLS")
#--------------------------------------------------------

"""Ridge Regression"""
print("d) Ridge analysis:")
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

    print("Lambda=",i)
    print("MSE =", mse_Ridge)
    print("R2-score =", r2score_Ridge)
    print("Variance =", var_Ridge)

lambda_params = [10, 0.1, 1e-3, 1e-6, 1e-10]
Z = FrankeFunction(X, Y) + noise
for j in lambda_params:
    #plot_conf_int(N, hyperparam=j, method="Ridge")

    Mse, R2, Var, Bias = k_fold_CV(X, Y, Z, folds=5, dim=5, hyperparam=j, method="Ridge", train=False)
    print("Lambda:", j)
    print("MSE k-fold =", Mse)
    print("R2-score k-fold =", R2)
    print("Variance k-fold=", Var)
    print("Bias k-fold =", Bias)

#fig_bias_var(X, Y, p=12, method="Ridge")   # Evt teste for forskjellig N...
    
#-------------------------------------------------------
"""Lasso Regression"""
print("e) Lasso analysis:")
Z = FrankeFunction(X, Y)
z = np.ravel(Z)
#hyperparam = 1e-10
lambda_Lasso = np.logspace(-10, 0, 9)
for i in lambda_Lasso:
    beta_Lasso = Lasso(M, z, hyperparam=i)  # lambda=0 should give ~OLS
    y_tilde = M @ beta_Lasso
    mse_Lasso = MSE(Z, y_tilde)
    r2score_Lasso = R2score(Z, y_tilde)
    var_Lasso = VAR(y_tilde)
    
    print("Lambda:", i)
    print("MSE =", mse_Lasso)
    print("R2-score =", r2score_Lasso)
    print("Variance =", var_Lasso)

Z = Z + noise
lambda_params = [0.1, 1e-3, 1e-6, 1e-10]
for j in lambda_params:
    #plot_conf_int(N, hyperparam=j, method="Lasso")    
    
    print("Lambda:", j)
    Mse, R2, Var, Bias = k_fold_CV(X, Y, Z, folds=5, dim=5, hyperparam=j, method="Lasso", train=False)
    print("MSE k-fold =", Mse)
    print("R2-score k-fold =", R2)
    print("Variance k-fold=", Var)
    print("Bias k-fold =", Bias)
    
#fig_bias_var(X, Y, p=12, method="Lasso")
