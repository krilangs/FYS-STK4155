import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import funcs as fun

np.random.seed(777)

n = 5

terrain = imread("n59_e010_1arc_v3.tif")[:-1, :-1]
z = terrain/np.max(terrain)
x = np.linspace(0,1, z.shape[0])
y = np.linspace(0,1, z.shape[1])
X, Y = np.meshgrid(x, y)
#print(z.shape)
#print(x.shape)
#print(y.shape)
M = fun.DesignMatrix(X, Y, n)

beta_OLS = fun.OLS(M, z)
y_tilde = M @ beta_OLS
mse_OLS = fun.MSE(z, y_tilde)
r2score_OLS = fun.R2score(z, y_tilde)
var_OLS = fun.VAR(y_tilde)
Bias_OLS = fun.BIAS(z, y_tilde)

print("MSE = %.10f" %mse_OLS)
print("R2-score = %.10f" %r2score_OLS)
print("Variance = %.10f" %var_OLS)
print("Bias = %.10f" %Bias_OLS)