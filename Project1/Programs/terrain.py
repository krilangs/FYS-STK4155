import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import funcs as fun
from plots import terrain_plot, seaborn_heatmap
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings

#n = 5

full_terrain = imread("n59_e010_1arc_v3.tif")

terrain_data = full_terrain[::30, ::30]

ny, nx = np.shape(terrain_data)

x_grid, y_grid = np.meshgrid(np.linspace(0,nx/max(nx,ny),nx), np.linspace(0,ny/max(nx,ny),ny))

x = np.ravel(x_grid)
y = np.ravel(y_grid)
z = np.ravel(terrain_data)

# OLS
min_degree = 8
max_degree = 20
degrees = np.linspace(min_degree, max_degree, (max_degree-min_degree)+1, dtype=int)
A = np.zeros((5, len(degrees)))
#for i, dims in enumerate(degrees):
    #Mse, R2, Var, Bias = fun.k_fold_CV_terrain(x, y, z, folds=5, dim=dims, hyperparam=1, method="OLS", Train=False)
    #A[:,i] = dims, Mse, R2, Var, Bias
    #print("Dimension = ", dims)
    #print("MSE k-fold = %.6f" %Mse)
    #print("R2-score k-fold = %.6f" %R2)
    #print("Variance k-fold = %.6f" %Var)
    #print("Bias k-fold = %.6f" %Bias)

#fun.make_tab(A, task="terrain_OLS", string="6f") 

#terrain_plot(x, y, z, p=18, plots="OLS")  

# Ridge
lambda_params = [1e-3, 1e-5, 1e-7, 1e-10, 1e-12]
dim_params = [5, 7, 8, 10, 11, 12]
R2 = np.zeros((len(dim_params),(len(lambda_params))))
#for i, dims in enumerate(dim_params):
    #for j, lambd in enumerate(lambda_params):        
        #Mse, R2[i][j], Var, Bias = fun.k_fold_CV_terrain(x, y, z, folds=5, dim=dims, hyperparam=lambd, method="Ridge", Train=False)

# Test best dimension (n) and model complexity (lambda)
#seaborn_heatmap(R2, lambda_params, dim_params, noise=None, method="Ridge")

#terrain_plot(x, y, z, p=max_degree, plots="Ridge")

# Lasso
lambda_params = [1e-3, 1e-5, 1e-7, 1e-10, 1e-12]
dim_params = [5, 6, 7, 8, 9, 10, 12]
R2 = np.zeros((len(dim_params),(len(lambda_params))))
for i, dims in enumerate(dim_params):
    for j, lambd in enumerate(lambda_params):
        Mse, R2[i][j], Var, Bias = fun.k_fold_CV_terrain(x, y, z, folds=5, dim=dims, hyperparam=lambd, method="Lasso", Train=False)

# Test best dimension (n) and model complexity (lambda)
seaborn_heatmap(R2, lambda_params, dim_params, noise=None, method="Lasso")
        
#terrain_plot(x, y, z, p=max_degree, plots="Lasso")
        
@ignore_warnings(category = ConvergenceWarning)
def best_params(x, y, z, n, hyperparam, method=""):
    Mse, R2, Var, Bias = fun.k_fold_CV_terrain(x, y, z, folds=5, dim=n, hyperparam=hyperparam, method=method, Train=False)
    #M = fun.DesignMatrix(x,y,n)
    #if method == "OLS":
        #beta = fun.OLS(M, z)
    #elif method == "Ridge":
        #beta = fun.Ridge(M, z, hyperparam)
    #elif method == "Lasso":
        #beta = fun.Lasso(M, z, hyperparam)   

    #y_tilde = M @ beta
    #mse = fun.MSE(terrain_data, y_tilde)
    #r2score = fun.R2score(terrain_data, y_tilde)
    #var = fun.VAR(y_tilde)
    #bias = fun.BIAS(terrain_data, y_tilde)
    if method == "OLS":
        print("Best params: n = " + str(n))
    else:
        print("Best params: n = " + str(n) + "and hyperparam = " + str(hyperparam))
    print("MSE = %.6f" %Mse)
    print("R2-score = %.6f" %R2)
    print("Variance = %.6f" %Var)
    print("Bias = %.6f" %Bias)
    if method == "OLS":
        A = [n, Mse, R2, Var, Bias]
    else:
        A = [hyperparam, n, Mse, R2, Var, Bias]
    #fun.make_tab(A, task="best_"+str(method)+"terrain", string="6f")
    #plot3d_terrain(X, Y, Z, z2, method="terrain")

if __name__ == "__main__":
    # OLS
    #best_params(x, y, z, n, hyperparam=1, method="OLS")
    # Ridge
    #best_params(x, y, z, n, hyperparam, method="Ridge")
    # Lasso
    #best_params(x, y, z, n, hyperparam, method="Lasso")
    
    pass