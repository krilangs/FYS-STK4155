import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import funcs as fun
from plots import terrain_plot, seaborn_heatmap

np.random.seed(777)

n = 5

full_terrain = imread("n59_e010_1arc_v3.tif")
U, S, V = np.linalg.svd(full_terrain)        #using SVD method to decompose image

def redel(A , r):
	C = np.zeros(r)
	for i in range(r):
    		for j in range(r):
        		C[i , j] = A[i , j]


y  = redel(S , r)              # redel as a function reduces dimension of S
Y  = np.diag(y1)               # Saving singular values of S1 into Y
Ynew = np.zeros(full_terrain.shape)     # Ynew is compressed version of img

for i in range(r):
    Y1new = Ynew + Y[i]* U[:,i]*V[:,i].T

"""
terrain_data = full_terrain[::20, ::20]

ny, nx = np.shape(terrain_data)

x_grid, y_grid = np.meshgrid(np.linspace(0,nx/max(nx,ny),nx), np.linspace(0,ny/max(nx,ny),ny))

x = np.ravel(x_grid)
y = np.ravel(y_grid)
z = np.ravel(terrain_data)

# OLS
min_degree = 4
max_degree = 15
degrees = np.linspace(min_degree, max_degree, (max_degree-min_degree)+1, dtype=int)
A = np.zeros((5, len(degrees)))
for i, dims in enumerate(degrees):
    Mse, R2, Var, Bias = fun.k_fold_CV_terrain(x, y, z, folds=5, dim=n, hyperparam=1, method="OLS", Train=False)
    A[:,i] = dims, Mse, R2, Var, Bias
    print("Dimension = ", dims)
    print("MSE k-fold = %.10f" %Mse)
    print("R2-score k-fold = %.10f" %R2)
    print("Variance k-fold = %.10f" %Var)
    print("Bias k-fold = %.10f" %Bias)

fun.make_tab(A, task="terrain_OLS", string="6f") 

#terrain_plot(x, y, z, p=max_degree, plots="OLS")  

# Ridge
lambda_params = [1e-3, 1e-5, 1e-7, 1e-10, 1e-12]
dim_params = [7, 8, 9, 10, 11, 12, 13]
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
#for i, dims in enumerate(dim_params):
    #for j, lambd in enumerate(lambda_params):
        #Mse, R2[i][j], Var, Bias = fun.k_fold_CV_terrain(x, y, z, folds=5, dim=dims, hyperparam=lambd, method="Lasso", Train=False)

# Test best dimension (n) and model complexity (lambda)
#seaborn_heatmap(R2, lambda_params, dim_params, noise=None, method="Lasso")
        
#terrain_plot(x, y, z, p=max_degree, plots="Lasso")

def best_params(n, hyperparam, method=""):
    M = fun.DesignMatrix(x,y,n)
    if method == "OLS":
        beta = fun.OLS(M, z)
    elif method == "Ridge":
        beta = fun.Ridge(M, z, hyperparam)
    elif method == "Lasso":
        beta = fun.Lasso(M, z, hyperparam)   

    y_tilde = M @ beta
    mse = fun.MSE(terrain_data, y_tilde)
    r2score = fun.R2score(terrain_data, y_tilde)
    var = fun.VAR(y_tilde)
    bias = fun.BIAS(terrain_data, y_tilde)
    print("Best params: n = " + n + "hyperparam = " + hyperparam)
    print("MSE = %.6f" %mse)
    print("R2-score = %.6f" %r2score)
    print("Variance = %.6f" %var)
    print("Bias = %.6f" %bias)
    if method == "OLS":
        A = [n, mse, r2score, var, bias]
    else:
        A = [hyperparam, n, mse, r2score, var, bias]
    #fun.make_tab(A, task="best_"+str(method)+"terrain", string="6f")
    #plot3d(X, Y, Z, z2, method="terrain")
"""