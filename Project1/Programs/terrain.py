from __future__ import division
import numpy as np
from imageio import imread
import funcs as fun
from plots import bias_var_terrain, seaborn_heatmap, plot3d_terrain
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings

full_terrain = imread("n59_e010_1arc_v3.tif")

terrain_data = full_terrain[::30, ::30]

ny, nx = np.shape(terrain_data)
X = np.linspace(0, nx/max(nx, ny), nx)
Y = np.linspace(0, ny/max(nx, ny), ny)

x_grid, y_grid = np.meshgrid(X, Y)

x = np.ravel(x_grid)
y = np.ravel(y_grid)
z = np.ravel(terrain_data)

def OLS_terrain():
    """Fit the OLS model to the terrain data, and find best parameters."""
    min_degree = 10
    max_degree = 22
    degrees = np.linspace(min_degree, max_degree, (max_degree-min_degree)+1, dtype=int)
    A = np.zeros((5, len(degrees)))
    for i, dims in enumerate(degrees):
        Mse, R2, Var, Bias = fun.k_fold_CV_terrain(x, y, z, folds=5, dim=dims,
                                                   hyperparam=1, method="OLS", Train=False)
        A[:,i] = dims, Mse, R2, Var, Bias
        print("Dimension = ", dims)
        print("MSE k-fold = %.6f" %Mse)
        print("R2-score k-fold = %.6f" %R2)
        print("Variance k-fold = %.6f" %Var)
        print("Bias k-fold = %.6f" %Bias)
    fun.make_tab(A, task="terrain_OLS", string="6f")

    #bias_var_terrain(x, y, z, p=20, plots="OLS")

#-----------------------------------------------------

def Ridge_terrain():
    """Fit the Ridge model to the terrain data, and find best parameters."""
    lambda_params = [1e-3, 1e-5, 1e-6, 1e-7, 1e-10]
    dim_params = [12, 13, 14, 15, 16, 17]
    R2 = np.zeros((len(dim_params), (len(lambda_params))))
    for i, dims in enumerate(dim_params):
        for j, lambd in enumerate(lambda_params):
            Mse, R2[i][j], Var, Bias = fun.k_fold_CV_terrain(x, y, z, folds=5,
                                       dim=dims, hyperparam=lambd, method="Ridge", Train=False)

    # Test best dimension (n) and model complexity (lambda)
    seaborn_heatmap(R2, lambda_params, dim_params, noise=None, method="Ridge")

    #bias_var_terrain(x, y, z, p=20, plots="Ridge")

#-----------------------------------------------------

@ignore_warnings(category=ConvergenceWarning)
def Lasso_terrain():
    """Fit the Lasso model to the terrain data, and find best parameters."""
    lambda_params = [1e-3, 1e-5, 1e-6, 1e-7, 1e-10]
    dim_params = [12, 13, 14, 15, 16, 18]
    R2 = np.zeros((len(dim_params), (len(lambda_params))))
    for i, dims in enumerate(dim_params):
        for j, lambd in enumerate(lambda_params):
            Mse, R2[i][j], Var, Bias = fun.k_fold_CV_terrain(x, y, z, folds=5,
                                       dim=dims, hyperparam=lambd, method="Lasso", Train=False)

    # Test best dimension (n) and model complexity (lambda)
    seaborn_heatmap(R2, lambda_params, dim_params, noise=None, method="Lasso")

    #bias_var_terrain(x, y, z, p=20, plots="Lasso")

#-----------------------------------------------------

@ignore_warnings(category=ConvergenceWarning)
def best_params(x, y, z, n, hyperparam, method=""):
    Mse, R2, Var, Bias = fun.k_fold_CV_terrain(x, y, z, folds=5, dim=n, hyperparam=hyperparam,
                                               method=method, Train=False)
    M = fun.DesignMatrix(x_grid, y_grid, n)
    if method == "OLS":
        beta = fun.OLS(M, z)
    elif method == "Ridge":
        beta = fun.Ridge(M, z, hyperparam)
    elif method == "Lasso":
        beta = fun.Lasso(M, z, hyperparam)

    y_tilde = M @ beta

    if method == "OLS":
        print("Best params: n = " + str(n))
    else:
        print("Best params: n = " + str(n) + " and hyperparam = " + str(hyperparam))
    print("MSE = %.6f" %Mse)
    print("R2-score = %.6f" %R2)
    print("Variance = %.6f" %Var)
    print("Bias = %.6f" %Bias)
    if method == "OLS":
        A = [n, Mse, R2, Var, Bias]
    else:
        A = [hyperparam, n, Mse, R2, Var, Bias]
    fun.make_tab(A, task="best_"+str(method)+"_terrain", string="6f")

    #plot3d_terrain(x_grid, y_grid, terrain_data, y_tilde, method)

if __name__ == "__main__":
    # OLS
    #OLS_terrain()
    #best_params(x, y, z, n=15, hyperparam=1, method="OLS")
    # Ridge
    #Ridge_terrain()
    #best_params(x, y, z, n=16, hyperparam=1e-5, method="Ridge")
    # Lasso
    #Lasso_terrain()
    best_params(x, y, z, n=16, hyperparam=1e-5, method="Lasso")

    pass
