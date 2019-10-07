import numpy as np
import funcs as fun
from plots import plot3d, plot_conf_int, fig_bias_var, seaborn_heatmap
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
np.random.seed(777)

#------------------------------------------------
def ex_a(N, n):
    """OLS on Franke function"""
    print("a) Ordinary Least Square:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    noise = 0.2*np.random.normal(0, 1, size=X.shape)
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

    print("MSE = %.6f" %mse_OLS)
    print("R2-score = %.6f" %r2score_OLS)
    print("Variance = %.6f" %var_OLS)
    print("Bias = %.6f" %Bias_OLS)

    #plot_conf_int(N, hyperparam=1, method="OLS")
    #plot3d(X, Y, Z, Zn, method=None)

#------------------------------------------------

def ex_b(N, n):
    """Resampling with k-fold OLS"""
    print("b) Resampling of test data with k-fold:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    noise = 0.2*np.random.normal(0, 1, size=X.shape)
    Z = fun.FrankeFunction(X, Y)
    Zn = fun.FrankeFunction(X, Y) + noise
    
    dim_params = [4, 5, 6, 7, 8, 9, 10]
    A = np.zeros((9, len(dim_params)))

    for i, dims in enumerate(dim_params):
        Mse, R2, Var, Bias = fun.k_fold_CV_franke(X, Y, Z, folds=5, dim=dims, hyperparam=1, method="OLS", train=False)
        Mse_n, R2_n, Var_n, Bias_n = fun.k_fold_CV_franke(X, Y, Zn, folds=5, dim=dims, hyperparam=1, method="OLS", train=False)
        A[:,i] = dims, Mse, Mse_n, R2, R2_n, Var, Var_n, Bias, Bias_n
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
    
    fun.make_tab(A, task="ex_b_OLS", string="6f")    

#------------------------------------------------

def ex_c(N):
    """Bias-variance tradeoff OLS"""
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
    noise = 0.2*np.random.normal(0, 1, size=X.shape)
    Z = fun.FrankeFunction(X, Y)
    z = np.ravel(Z)
    Zn = fun.FrankeFunction(X, Y) + noise
    M = fun.DesignMatrix(X, Y, n)
    lambda_Ridge = np.logspace(-7, 1, 9)

    A = np.zeros((5, len(lambda_Ridge)))
    # Without resampling and noise
    #for j, i in enumerate(lambda_Ridge): 
        #beta_Ridge = fun.Ridge(M, z, hyperparam=i)
        #y_tilde = M @ beta_Ridge
        #mse_Ridge = fun.MSE(Z, y_tilde)
        #r2score_Ridge = fun.R2score(Z, y_tilde)
        #var_Ridge = fun.VAR(y_tilde)
        #bias_Ridge = fun.BIAS(Z, y_tilde)
        #A[:,j] = i, mse_Ridge, r2score_Ridge, var_Ridge, bias_Ridge

        #print("Lambda=",i)
        #print("MSE =", mse_Ridge)
        #print("R2-score =", r2score_Ridge)
        #print("Variance =", var_Ridge)
        #print("Bias = ", bias_Ridge)
    #fun.make_tab(A, task="ex_d_Ridge", string="5e") 
    
    # With resampling
    lambda_params = [ 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10]
    dim_params = [5, 6, 7, 8, 9, 10, 11]
    R2 = np.zeros((len(dim_params),(len(lambda_params))))
    R2_n = np.zeros_like(R2)
    #for i, dims in enumerate(dim_params):
        #for j, lambd in enumerate(lambda_params):        
            #Mse, R2[i][j], Var, Bias = fun.k_fold_CV_franke(X, Y, Z, folds=5, dim=dims, hyperparam=lambd, method="Ridge", train=False)
            #Mse_n, R2_n[i][j], Var_n, Bias_n = fun.k_fold_CV_franke(X, Y, Zn, folds=5, dim=dims, hyperparam=lambd, method="Ridge", train=False)
        
        
    # Test best dimension (n) and model complexity (lambda)
    #seaborn_heatmap(R2, lambda_params, dim_params, noise="No", method="Ridge")
    #seaborn_heatmap(R2_n, lambda_params, dim_params, noise="Yes", method="Ridge")
    
    # Use best dimension parameter
    B = np.zeros((9, len(lambda_params)))
    #for j, lambd in enumerate(lambda_params):
        #plot_conf_int(N, hyperparam=lambd, method="Ridge")

        #Mse, R2, Var, Bias = fun.k_fold_CV_franke(X, Y, Z, folds=5, dim=n, hyperparam=lambd, method="Ridge", train=False)
        #Mse_n, R2_n, Var_n, Bias_n = fun.k_fold_CV_franke(X, Y, Zn, folds=5, dim=n, hyperparam=lambd, method="Ridge", train=False)
        #B[:,j] = lambd, Mse, Mse_n, R2, R2_n, Var, Var_n, Bias, Bias_n

        #print("Franke function without noise:")
        #print("Lambda:", lambd)
        #print("MSE k-fold = %.10f" %Mse)
        #print("R2-score k-fold = %.10f" %R2)
        #print("Variance k-fold = %.10f" %Var)
        #print("Bias k-fold = %.10f" %Bias)
        #print("Franke function with noise:")
        #print("Lambda:", lambd)
        #print("MSE k-fold = %.10f" %Mse_n)
        #print("R2-score k-fold = %.10f" %R2_n)
        #print("Variance k-fold = %.10f" %Var_n)
        #print("Bias k-fold = %.10f" %Bias_n)

    #fun.make_tab(B, task="ex_d_kfold", string="5e")

    fig_bias_var(X, Y, p=12, method="Ridge")
    
#-------------------------------------------------------

@ignore_warnings(category = ConvergenceWarning)
def ex_e(N, n):
    """Lasso Regression"""
    print("e) Lasso analysis:")
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    noise = 0.2*np.random.normal(0, 1, size=X.shape)
    Z = fun.FrankeFunction(X, Y)
    z = np.ravel(Z)
    Zn = fun.FrankeFunction(X, Y) + noise
    
    M = fun.DesignMatrix(X, Y, n)

    lambda_Lasso = np.logspace(-10, 0, 9)
    A = np.zeros((5, len(lambda_Lasso)))
    # Without resampling and noise
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
    
    # With resampling
    lambda_params = [1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12]
    dim_params = [5, 6, 7, 8, 9, 10, 12]
    R2 = np.zeros((len(dim_params),(len(lambda_params))))
    R2_n = np.zeros_like(R2)
    #for i, dims in enumerate(dim_params):
        #for j, lambd in enumerate(lambda_params):
            #Mse, R2[i][j], Var, Bias = fun.k_fold_CV_franke(X, Y, Z, folds=5, dim=dims, hyperparam=lambd, method="Lasso", train=False)
            #Mse_n, R2_n[i][j], Var_n, Bias_n = fun.k_fold_CV_franke(X, Y, Zn, folds=5, dim=dims, hyperparam=lambd, method="Lasso", train=False)

    # Test best dimension (n) and model complexity (lambda)
    #seaborn_heatmap(R2, lambda_params, dim_params, noise="No", method="Lasso")
    #seaborn_heatmap(R2_n, lambda_params, dim_params, noise="Yes", method="Lasso")        


    B = np.zeros((9, len(lambda_params)))
    #for j, lambd in enumerate(lambda_params):
        #plot_conf_int(N, hyperparam=lambd, method="Lasso")

        #Mse, R2, Var, Bias = fun.k_fold_CV_franke(X, Y, Z, folds=5, dim=n, hyperparam=lambd, method="Lasso", train=False)
        #Mse_n, R2_n, Var_n, Bias_n = fun.k_fold_CV_franke(X, Y, Zn, folds=5, dim=n, hyperparam=lambd, method="Lasso", train=False)
        #B[:,j] = lambd, Mse, Mse_n, R2, R2_n, Var, Var_n, Bias, Bias_n
        
        #print("Franke function without noise:")
        #print("Lambda:", lambd)
        #print("MSE k-fold = %.10f" %Mse)
        #print("R2-score k-fold = %.10f" %R2)
        #print("Variance k-fold = %.10f" %Var)
        #print("Bias k-fold = %.10f" %Bias)
        #print("Franke function with noise:")
        #print("Lambda:", lambd)
        #print("MSE k-fold = %.10f" %Mse_n)
        #print("R2-score k-fold = %.10f" %R2_n)
        #print("Variance k-fold = %.10f" %Var_n)
        #print("Bias k-fold = %.10f" %Bias_n)

    #fun.make_tab(B, task="ex_e_kfold", string="5e")
    fig_bias_var(X, Y, p=12, method="Lasso")

#-------------------------------------------------------

@ignore_warnings(category = ConvergenceWarning)
def best(N, n, hyperparam, method=""):
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    X, Y = np.meshgrid(x, y)
    noise = 0.2*np.random.normal(0, 1, size=X.shape)
    
    Zn = fun.FrankeFunction(X, Y) + noise
    z = np.ravel(Zn)
    
    M = fun.DesignMatrix(X, Y, n)
    if method == "OLS":
        beta = fun.OLS(M, z)
    elif method == "Ridge":
        beta = fun.Ridge(M, z, hyperparam)
    elif method == "Lasso":
        beta = fun.Lasso(M, z, hyperparam)
        
    y_tilde = M @ beta
    mse = fun.MSE(Zn, y_tilde)
    r2score = fun.R2score(Zn, y_tilde)
    var = fun.VAR(y_tilde)
    bias = fun.BIAS(Zn, y_tilde)
    print("MSE = %.6f" %mse)
    print("R2-score = %.6f" %r2score)
    print("Variance = %.6f" %var)
    print("Bias = %.6f" %bias)
    if method == "OLS":
        A = [n, mse, r2score, var, bias]
    else:
        A = [hyperparam, n, mse, r2score, var, bias]

    fun.make_tab(A, task="best_"+str(method), string="6f")
    plot3d(X, Y, Zn, y_tilde, method)
    
if __name__ == "__main__":
    #ex_a(N=50, n=5)
    #ex_b(N=50, n=5)
    #ex_c(N=50)
    #ex_d(N=50, n=5)
    #ex_e(N=50, n=5)
    #best(N=50, n=9, hyperparam=1e-7, method="Lasso")
    pass