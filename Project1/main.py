import numpy as np
from sklearn.model_selection import train_test_split
import funcs as fun
from plots import plot3d, plot_conf_int, fig_bias_var

np.random.seed(777)
"""
N = 40
n = 5

x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))

X, Y = np.meshgrid(x, y)
noise = 0.8*np.random.normal(0, 1, size=X.shape)
Z = fun.FrankeFunction(X, Y)
z = np.ravel(Z)
Zn = fun.FrankeFunction(X, Y) + noise
zn = np.ravel(Zn)

M = fun.DesignMatrix(X, Y, n)
"""
#------------------------------------------------
def ex_a(N, n):
    """OLS on Franke function"""
    print("a) Ordinary Least Square:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    noise = 0.8*np.random.normal(0, 1, size=X.shape)
    Z = fun.FrankeFunction(X, Y)
    z = np.ravel(Z)
    Zn = fun.FrankeFunction(X, Y) + noise

    M = fun.DesignMatrix(X, Y, n)

    beta_OLS = fun.OLS(M, z)
    y_tilde = M @ beta_OLS
    mse_OLS = fun.MSE(Z, y_tilde)
    r2score_OLS = fun.R2score(Z, y_tilde)
    var_OLS = fun.VAR(y_tilde)
    Bias_OLS = fun.BIAS(Z, y_tilde)

    print("MSE = %.10f" %mse_OLS)
    print("R2-score = %.10f" %r2score_OLS)
    print("Variance = %.10f" %var_OLS)
    print("Bias = %.10f" %Bias_OLS)

    #plot_conf_int(N=100, hyperparam=1, method="OLS")
    #plot3d(X, Y, Z, Zn)

#------------------------------------------------

def ex_b(N, n):
    """Resampling with k-fold"""
    print("b) Resampling of test data with k-fold:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    noise = 0.8*np.random.normal(0, 1, size=X.shape)
    Z = fun.FrankeFunction(X, Y)
    Zn = fun.FrankeFunction(X, Y) + noise
    
    Mse, R2, Var, Bias = fun.k_fold_CV(X, Y, Z, folds=5, dim=n, hyperparam=1, method="OLS", train=False)
    Mse_n, R2_n, Var_n, Bias_n = fun.k_fold_CV(X, Y, Zn, folds=5, dim=n, hyperparam=1, method="OLS", train=False)
    print("Franke function without noise:")
    print("MSE k-fold = %.10f" %Mse)
    print("R2-score k-fold = %.10f" %R2)
    print("Variance k-fold = %.10f" %Var)
    print("Bias k-fold = %.10f" %Bias)
    print("Franke function with noise:")
    print("MSE k-fold = %.10f" %Mse_n)
    print("R2-score k-fold = %.10f" %R2_n)
    print("Variance k-fold = %.10f" %Var_n)
    print("Bias k-fold = %.10f" %Bias_n)

    A = np.array([[Mse, Mse_n],[R2, R2_n],[Var, Var_n],[Bias, Bias_n]])
    #fun.make_tab(A, task="ex_b_OLS", string="6f")    

#------------------------------------------------

def ex_c(N):
    """Bias-variance tradeoff"""
    print("c) Plotting bias-variance tradeoff:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    fig_bias_var(X, Y, p=11, method="OLS")

#------------------------------------------------

def ex_d(N, n):
    """Ridge Regression"""
    print("d) Ridge analysis:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    noise = 0.8*np.random.normal(0, 1, size=X.shape)
    Z = fun.FrankeFunction(X, Y)
    z = np.ravel(Z)
    Zn = fun.FrankeFunction(X, Y) + noise
    M = fun.DesignMatrix(X, Y, n)
    lambda_Ridge = np.logspace(-7, 1, 9)

    A = np.zeros((5, len(lambda_Ridge)))
    
    for j, i in enumerate(lambda_Ridge): 
        beta_Ridge = fun.Ridge(M, z, hyperparam=i)
        y_tilde = M @ beta_Ridge
        mse_Ridge = fun.MSE(Z, y_tilde)
        r2score_Ridge = fun.R2score(Z, y_tilde)
        var_Ridge = fun.VAR(y_tilde)
        bias_Ridge = fun.BIAS(Z, y_tilde)
        A[:,j] = i, mse_Ridge, r2score_Ridge, var_Ridge, bias_Ridge

        print("Lambda=",i)
        print("MSE =", mse_Ridge)
        print("R2-score =", r2score_Ridge)
        print("Variance =", var_Ridge)
        print("Bias = ", bias_Ridge)

    #fun.make_tab(A, task="ex_d_Ridge", string="5e") 
    lambda_params = [10, 0.1, 1e-3, 1e-6, 1e-10]
    B = np.zeros((9, len(lambda_params)))

    for j, i in enumerate(lambda_params):
        #plot_conf_int(N, hyperparam=j, method="Ridge")
        
        Mse, R2, Var, Bias = fun.k_fold_CV(X, Y, Z, folds=5, dim=5, hyperparam=i, method="Ridge", train=False)
        Mse_n, R2_n, Var_n, Bias_n = fun.k_fold_CV(X, Y, Zn, folds=5, dim=5, hyperparam=i, method="Ridge", train=False)
        B[:,j] = i, Mse, Mse_n, R2, R2_n, Var, Var_n, Bias, Bias_n
        
        print("Franke function without noise:")
        print("Lambda:", i)
        print("MSE k-fold = %.10f" %Mse)
        print("R2-score k-fold = %.10f" %R2)
        print("Variance k-fold = %.10f" %Var)
        print("Bias k-fold = %.10f" %Bias)
        print("Franke function with noise:")
        print("Lambda:", i)
        print("MSE k-fold = %.10f" %Mse_n)
        print("R2-score k-fold = %.10f" %R2_n)
        print("Variance k-fold = %.10f" %Var_n)
        print("Bias k-fold = %.10f" %Bias_n)
        
    #fun.make_tab(B, task="ex_d_kfold", string="5e") 
    #fig_bias_var(X, Y, p=12, method="Ridge")
    
#-------------------------------------------------------

def ex_e(N, n):
    """Lasso Regression"""
    print("e) Lasso analysis:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    noise = 0.8*np.random.normal(0, 1, size=X.shape)
    Z = fun.FrankeFunction(X, Y)
    z = np.ravel(Z)
    Zn = fun.FrankeFunction(X, Y) + noise
    
    M = fun.DesignMatrix(X, Y, n)

    lambda_Lasso = np.logspace(-10, 0, 9)
    A = np.zeros((5, len(lambda_Lasso)))
    
    #for j, i in enumerate(lambda_Lasso):
        #beta_Lasso = fun.Lasso(M, z, hyperparam=i)  # lambda=0 should give ~OLS
        #y_tilde = M @ beta_Lasso
        #mse_Lasso = fun.MSE(Z, y_tilde)
        #r2score_Lasso = fun.R2score(Z, y_tilde)
        #var_Lasso = fun.VAR(y_tilde)
        #bias_Lasso = fun.BIAS(Z, y_tilde)
        #A[:,j] = i, mse_Lasso, r2score_Lasso, var_Lasso, bias_Lasso
    
        #print("Lambda:", i)
        #print("MSE =", mse_Lasso)
        #print("R2-score =", r2score_Lasso)
        #print("Variance =", var_Lasso)
        #print("Bias = ", bias_Lasso)

    #fun.make_tab(A, task="ex_e_Lasso", string="5e") 

    lambda_params = [0.1, 1e-3, 1e-6, 1e-10]
    B = np.zeros((9, len(lambda_params)))

    #for j, i in enumerate(lambda_params):
        #plot_conf_int(N, hyperparam=i, method="Lasso")    
        
        #Mse, R2, Var, Bias = fun.k_fold_CV(X, Y, Z, folds=5, dim=5, hyperparam=i, method="Lasso", train=False)
        #Mse_n, R2_n, Var_n, Bias_n = fun.k_fold_CV(X, Y, Zn, folds=5, dim=5, hyperparam=i, method="Lasso", train=False)

        #B[:,j] = i, Mse, Mse_n, R2, R2_n, Var, Var_n, Bias, Bias_n
        #print("Franke function without noise:")
        #print("Lambda:", i)
        #print("MSE k-fold = %.10f" %Mse)
        #print("R2-score k-fold = %.10f" %R2)
        #print("Variance k-fold = %.10f" %Var)
        #print("Bias k-fold = %.10f" %Bias)
        #print("Franke function with noise:")
        #print("Lambda:", i)
        #print("MSE k-fold = %.10f" %Mse_n)
        #print("R2-score k-fold = %.10f" %R2_n)
        #print("Variance k-fold = %.10f" %Var_n)
        #print("Bias k-fold = %.10f" %Bias_n)

    #fun.make_tab(B, task="ex_e_kfold", string="5e")
    fig_bias_var(X, Y, p=12, method="Lasso")


if __name__ == "__main__":
    #ex_a(N=40, n=5)
    #ex_b(N=40, n=5)
    #ex_c(N=50)
    #ex_d(N=50, n=5)
    ex_e(N=40, n=5)

    pass