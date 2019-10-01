"""
File containing functions that create plots,
and which are imported to the main file.
"""
from mpl_toolkits.mplot3d import Axes3D
from imageio import imread
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

np.random.seed(1337)

def plot3d(x, y, z, z2):
    """
    Function to make a 3d plot of the Franke function by using OLS.
    """
    path = "figures/"
    
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.text2D(0.05, 0.95, "3D plot of FrankeFunction", transform=ax.transAxes, size=16)
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(path + "Franke_3Dplot" + ".png")

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.text2D(0.05, 0.95, "3D plot of FrankeFunction with noise",
              transform=ax.transAxes, size=16)
    # Plot the surface.
    surf = ax.plot_surface(x, y, z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(path + "Franke_noise_3Dplot" + ".png")
    plt.show()

def plot_conf_int(N, method=""):
    """
    Function to plot the confidence interval of beta.
    """
    from proj1 import FrankeFunction, DesignMatrix, confidence_int, OLS
    
    fsize = 10 			# universal fontsize for plots
    path = "figures/"

    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    x, y = np.meshgrid(x, y)
    Z = np.ravel(FrankeFunction(x, y))
    Zn = Z + 0.8*np.random.normal(loc=0.0, scale=1, size=N*N)
    x = np.ravel(x)
    y = np.ravel(y)
    M = DesignMatrix(x, y, n=5)

    if method == "OLS":
        betaSTD_f = confidence_int(x, y, Z, method)
        beta_f = OLS(M, Z)
        betaSTD_z = confidence_int(x, y, Zn, method)
        beta_z = OLS(M, Zn)
        N = len(betaSTD_z)

    colors = ["mediumblue","crimson"]
    plt.plot(-10, -1, color=colors[0], label="without noise")
    plt.plot(-10, -1, color=colors[1], label="with noise")
    plt.legend()
    
    for i in range(N):
        plt.errorbar(i, beta_f[i], yerr=betaSTD_f[i], capsize=4, \
			color=colors[0], marker='.', markersize=7, elinewidth=2,\
			alpha=0.5)
        plt.errorbar(i, beta_z[i], yerr=betaSTD_z[i], capsize=4, \
			color=colors[1], marker='.', markersize=7, elinewidth=2,\
			alpha=0.5)
    xticks = [r'$\beta_{%d}$'%i for i in range(N)]
    plt.xticks(range(N), xticks, fontsize=fsize)
    plt.xlim(-1, N)
    plt.title("Confidence interval of $\\beta$ for " + str(method))
    plt.tight_layout()
    plt.savefig(path + "confIntBeta_" + str(method) + ".png")
    plt.grid("on")
    plt.show()

def fig_bias_var(x, y, hyperparam, p=10, reg=""):
    """
    Function to plot the test and training errors as
    functions of model complexity (p).
    """
    from proj1 import FrankeFunction, k_fold_CV
    
    complexity = np.arange(0, p+1)
    
    pred_error = np.zeros_like(complexity, dtype=float)
    p_e_noise = np.zeros_like(pred_error)
    #error_R2 = np.zeros_like(error_MSE)
    #error_var = np.zeros_like(error_MSE)
    #error_bias = np.zeros_like(error_MSE)
    
    pred_error_train = np.zeros_like(pred_error)
    p_e_t_noise = np.zeros_like(pred_error)
    #error_R2_train = np.zeros_like(error_MSE)
    #error_var_train = np.zeros_like(error_MSE)
    #error_bias_train = np.zeros_like(error_MSE)    
    
    Z = FrankeFunction(x, y)
    Zn = FrankeFunction(x, y) + 0.8*np.random.normal(0, 1, size=x.shape)
    for j in complexity:
        print(j)
        if reg == "OLS":
            pred_error[j], pred_error_train[j] = k_fold_CV(x, y, Z, folds=5, 
                                                  dim=j, hyperparam=1, 
                                                  reg="OLS", train=True)
            p_e_noise[j], p_e_t_noise[j] = k_fold_CV(x, y, Zn, folds=5, 
                                             dim=j, hyperparam=1, 
                                             reg="OLS", train=True)

    # Plot for FrankeFunction
    pred_log = np.log10(pred_error)
    pred_log_train = np.log10(pred_error_train)
    fig, ax = plt.subplots()
    plt.title("Bias-variance tradeoff train and test data\n "+ str(reg))
    fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
    ax.plot(complexity, pred_log_train, label="Train", color="g")
    ax.plot(complexity, pred_log, linestyle="--", label="Test", color="r")
    ax.set_xlabel("Model Complexity [polynomial degree]")
    ax.set_xticks(complexity[::2])
    ax.set_ylabel(r"log$_{10}$(Prediction Error)")
    ax.set_ylim([np.min(pred_log_train) - np.min(np.abs(pred_log_train)) * 0.1,
                 np.max(pred_log) + np.max(np.abs(pred_log)) * 0.3])

    ax.text(0.05, 0.75, "High bias\nLow variance\n<------",
            horizontalalignment="left",
            verticalalignment="baseline",
            transform=ax.transAxes)
    ax.text(0.95, 0.75, "Low bias\nHigh variance\n------>",
            horizontalalignment="right",
            verticalalignment="baseline",
            transform=ax.transAxes)

    ax.legend(loc=3)
    fig.tight_layout()
    fig.savefig("figures/biasvariancetradeoff_"+str(reg)+"_Franke.png", dpi=1000)
    plt.show()
    
    # Plot for FrankeFunction with noise
    pred_log = np.log10(p_e_noise)
    pred_log_train = np.log10(p_e_t_noise)
    fig, ax = plt.subplots()
    plt.title("Bias-variance tradeoff train and test data w/noise\n "+ str(reg))
    fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
    ax.plot(complexity, pred_log_train, label="Train", color="g")
    ax.plot(complexity, pred_log, linestyle="--", label="Test", color="r")
    ax.set_xlabel("Model Complexity [polynomial degree]")
    ax.set_xticks(complexity[::2])
    ax.set_ylabel(r"log$_{10}$(Prediction Error)")
    ax.set_ylim([np.min(pred_log_train) - np.min(np.abs(pred_log_train)) * 0.1,
                 np.max(pred_log) + np.max(np.abs(pred_log)) * 0.3])

    ax.text(0.05, 0.75, "High bias\nLow variance\n<------",
            horizontalalignment="left",
            verticalalignment="baseline",
            transform=ax.transAxes)
    ax.text(0.9, 0.75, "Low bias\nHigh variance\n------>",
            horizontalalignment="right",
            verticalalignment="baseline",
            transform=ax.transAxes)

    ax.legend(loc=3)
    fig.tight_layout()
    fig.savefig("figures/biasvariancetradeoff_"+str(reg)+"_Franke_noise.png", dpi=1000)
    plt.show()
        #elif reg == "Ridge":
            #beta = Ridge(X_train, Z_train, hyperparam)
        #elif reg == "Lasso":
            #beta = Lasso(X_train, Z_train, hyperparam)
            # Test data
            #beta_OLS = OLS(X_train, Z_train)
            #beta_Ridge = Ridge(X_train, Z_train, lamb=0.1)
            #beta_k, _, _ = k_fold_CV(X_train, Z_train, 5, shuffle = False)
            #beta_Lasso = Lasso(X_train, Z_train, alpha=0.000001)
            
        #z_tilde = X_test @ beta
            #z_tilde_Ridge = X_test @ beta_Ridge
            #z_tilde_k = X_test @ beta_k
            #z_tilde_Lasso = X_test @ beta_Lasso
            
        #error_MSE[0, j] += MSE(Z_test, z_tilde)
            #error_MSE[1, j] += MSE(Z_test, z_tilde_k)
            #error_MSE[2, j] += MSE(Z_test, z_tilde_Ridge)
            #error_MSE[3, j] += MSE(Z_test, z_tilde_Lasso)
        #error_R2[0, j] += R2score(Z_test, z_tilde)
            #error_R2[1, j] += R2score(Z_test, z_tilde_k)
            #error_R2[2, j] += R2score(Z_test, z_tilde_Ridge)
            #error_R2[3, j] += R2score(Z_test, z_tilde_Lasso)
        #error_var[0, j] += Var(z_tilde)
        #error_bias[0, j] += Bias(z_tilde, Z_test)
            
            # Training data
        #z_tilde = X_train @ beta
            #z_tilde_k = X_train @ beta_k
            #z_tilde_Ridge = X_train @ beta_Ridge
            #z_tilde_Lasso = X_train @ beta_Lasso
            
        #error_MSE_train[0, j] += MSE(Z_train, z_tilde)
            #error_MSE_train[1, j] += MSE(Z_train, z_tilde_k)
            #error_MSE_train[2, j] += MSE(Z_train, z_tilde_Ridge)
            #error_MSE_train[3, j] += MSE(Z_train, z_tilde_Lasso)
        #error_R2_train[0, j] += R2score(Z_train, z_tilde)
            #error_R2_train[1, j] += R2score(Z_train, z_tilde_k)
            #error_R2_train[2, j] += R2score(Z_train, z_tilde_Ridge)
            #error_R2_train[3, j] += R2score(Z_train, z_tilde_Lasso)
        #error_var_train[0, j] += Var(z_tilde)
        #error_bias_train[0, j] += Bias(z_tilde, Z_test)
            
        #total_test[j] = error_bias_train[0, j] + error_var_train[0, j] - error_MSE_train[0, j]
        #total_test[j] = error_bias[0, j] + error_var[0, j] - error_MSE[0, j]
            #print("Bias + Var - MSE=", error_bias[0, j] + error_var[0, j] - error_MSE[0, j])
    #error_MSE /= n
    ##error_R2 /= n
    #error_var /= n
    #error_MSE_train /= n
    #error_R2_train /= n
    #error_var_train /= n

    #plt.title('OLS - MSE')
    #plt.figure()
    #plt.plot(complexity, error_MSE[0], label = 'MSE:Test')
    #plt.plot(complexity, error_var[0], label = 'Var:Test')
    #plt.plot(complexity, error_bias[0], label = 'Bias:Test')
    #plt.plot(complexity, error_MSE_train[0], label = 'MSE:Training')
    #plt.ylim([0, np.max(error_MSE[0]*1.2)])
    #plt.legend()
    #plt.figure()
    #plt.title('OLS - Var')
    #plt.plot(complexity, error_var[0], label = 'Var:Test')
    #plt.plot(complexity, error_MSE_train[0], label = 'MSE:Training')
    #plt.plot(complexity, error_var_train[0], label = 'Var:Training')
    #plt.plot(complexity, error_bias_train[0], label = 'Bias:Training')
    #plt.ylim([0, np.max(error_MSE[0]*1.2)])
    #plt.legend()
    
    #plt.figure()
    #plt.plot(complexity, total_test, "r")
    #plt.plot(complexity, total_train)
    
    #plt.title('OLS - Bias')
    #plt.plot(complexity, error_bias[0], label = 'Bias:Test')
    #plt.plot(complexity, error_bias_train[0], label = 'Bias:Training')
    #plt.ylim([0, np.max(error_MSE[0]*1.2)])
    #plt.legend()
    """
    plt.title('k-fold')
    plt.plot(complexity, error_MSE[1], label = 'Test')
    plt.plot(complexity, error_MSE_train[1], label = 'Training')
    plt.ylim([0, np.max(error_MSE[1]*1.2)])

    plt.title('Ridge')
    plt.plot(complexity, error_MSE[2], label = 'Test')
    plt.plot(complexity, error_MSE_train[2], label = 'Training')
    plt.ylim([0, np.max(error_MSE[2]*1.2)])
    plt.legend()

    plt.title('Lasso')
    plt.plot(complexity, error_MSE[3], label = 'Test')
    plt.plot(complexity, error_MSE_train[3], label = 'Training')
    plt.ylim([0, np.max(error_MSE[3]*1.2)])
    plt.legend()
    """

def terrain():
    """
    Function that loads the terrain data of Hardangervidda, and shows it.
    """
    # Load the terrain
    vidda = imread("n59_e010_1arc_v3.tif")
    #print(terrain1.shape)
    # Show the terrain
    plt.figure()
    plt.title("Terrain over Hardangervidda")
    plt.imshow(vidda, cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # Show the terrain
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    
    #get the number of points
    n_y, n_x = np.shape(vidda)
    
    #making an x and y grid
    x_grid, y_grid = np.meshgrid(np.arange(n_x),np.arange(n_y))
    
    #print(np.shape(vidda))
    #print(np.shape(x_grid))
    
    # Plot the surface.
    surf = ax.plot_surface(x_grid, y_grid, vidda, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha = 0.5)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
#terrain()
