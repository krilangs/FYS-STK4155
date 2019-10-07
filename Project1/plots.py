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
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings

np.random.seed(777)
path = "figures/"

def plot3d(X, Y, Z, z2, method=""):
    """
    Function to make a 3d plot of the Franke function by using OLS.
    """
    if method == None:
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.text2D(0.05, 0.95, "3D plot of FrankeFunction", transform=ax.transAxes, size=16)
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        # Add a color bar which maps values to colors
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.tight_layout()
        plt.savefig(path + "Franke_3Dplot" + ".png")
        
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.text2D(0.05, 0.95, "3D plot of FrankeFunction with noise",
                  transform=ax.transAxes, size=16)
        # Plot the surface.
        surf = ax.plot_surface(X, Y, z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.tight_layout()
        plt.savefig(path + "Franke_noise_3Dplot" + ".png")
        
    elif method == "terrain":
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        
        
        
        
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.tight_layout()
        plt.savefig("terrain_figures/3Dplot_" +str(method)+ ".png")
        
        
    else:
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        x = np.ravel(X)
        y = np.ravel(Y)

        ax.scatter(x[::1], y[::1], z2[::1], s=2, color="black")
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.4)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(elev=20, azim=70)
        fig.tight_layout()
        plt.savefig(path + "3Dplot_best_" +str(method)+ ".png")

    plt.show()

@ignore_warnings(category = ConvergenceWarning)
def plot_conf_int(N, hyperparam, method=""):
    """
    Function to plot the confidence interval of beta.
    """
    from funcs import FrankeFunction, DesignMatrix, confidence_int, OLS, Ridge, Lasso
    
    fsize = 10 			# universal fontsize for plots

    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    x, y = np.meshgrid(x, y)
    Z = np.ravel(FrankeFunction(x, y))
    Zn = Z + 0.2*np.random.normal(loc=0.0, scale=1, size=N*N)
    x = np.ravel(x)
    y = np.ravel(y)
    M = DesignMatrix(x, y, n=5)

    if method == "OLS":
        betaSTD_f = confidence_int(x, y, Z, hyperparam, method)
        beta_f = OLS(M, Z)
        betaSTD_z = confidence_int(x, y, Zn, hyperparam, method)
        beta_z = OLS(M, Zn)
    elif method == "Ridge":
        betaSTD_f = confidence_int(x, y, Z, hyperparam, method)
        beta_f = Ridge(M, Z, hyperparam)
        betaSTD_z = confidence_int(x, y, Zn, hyperparam, method)
        beta_z = Ridge(M, Zn, hyperparam)
    elif method == "Lasso":
        betaSTD_f = confidence_int(x, y, Z, hyperparam, method)
        beta_f = Lasso(M, Z, hyperparam)
        betaSTD_z = confidence_int(x, y, Zn, hyperparam, method)
        beta_z = Lasso(M, Zn, hyperparam)
        
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
    if method == "Ridge" or method == "Lasso":
        plt.title("Confidence interval of $\\beta$ for " + str(method) + " and $\\lambda$=" + str(hyperparam))
    else:
        plt.title("Confidence interval of $\\beta$ for " + str(method))
    plt.tight_layout()
    plt.grid("on")
    if method == "Ridge" or method == "Lasso":
        plt.savefig(path + "confIntBeta_" + str(method) + "_" + str(hyperparam) + ".png")
    else:
        plt.savefig(path + "confIntBeta_" + str(method) + ".png")

    plt.show()

@ignore_warnings(category = ConvergenceWarning)
def fig_bias_var(x, y, p, method=""):
    """
    Function to plot the test and training errors as
    functions of model complexity (p).
    """
    from funcs import FrankeFunction, k_fold_CV_franke
    
    Z = FrankeFunction(x, y)
    Zn = FrankeFunction(x, y) + 0.2*np.random.normal(0, 1, size=x.shape)

    if method == "OLS":
        complexity = np.arange(0, p+1)
        pred_error = np.zeros_like(complexity, dtype=float)
        p_e_noise = np.zeros_like(pred_error)
        error_var = np.zeros_like(pred_error)
        error_bias = np.zeros_like(pred_error)
        pred_error_train = np.zeros_like(pred_error)
        p_e_t_noise = np.zeros_like(pred_error) 
        for j in complexity:
            print(j)
            pred_error[j], pred_error_train[j] = k_fold_CV_franke(x, y, Z, folds=5, 
                                                  dim=j, hyperparam=1, 
                                                  method="OLS", train=True)
            p_e_noise[j], p_e_t_noise[j] = k_fold_CV_franke(x, y, Zn, folds=5, 
                                             dim=j, hyperparam=1, 
                                             method="OLS", train=True)
            # Calculate variance and Bias for test data
            _, _, error_var[j], error_bias[j] = k_fold_CV_franke(x, y, Zn, folds=5, 
                                                   dim=j, hyperparam=1, 
                                                   method="OLS", train=False)
        
        # Plot for FrankeFunction
        fig, ax = plt.subplots()
        plt.title("Bias-variance tradeoff train and test data\n "+str(method))
        fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
        ax.plot(complexity, pred_error_train, label="Train", color="g")
        ax.plot(complexity, pred_error, linestyle="--", label="Test", color="r")
        ax.set_xlabel("Model Complexity [polynomial degree]")
        ax.set_xticks(complexity[::2])
        ax.set_ylabel("Prediction Error")

        ax.text(0.1, 0.75, "High bias\nLow variance\n<------",
                horizontalalignment="left",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.text(0.95, 0.75, "Low bias\nHigh variance\n------>",
                horizontalalignment="right",
                verticalalignment="baseline",
                transform=ax.transAxes)

        ax.legend(loc=9)
        fig.tight_layout()
        fig.savefig("figures/biasvariancetradeoff_"+str(method)+"_Franke.png", dpi=1000)
    
        # Plot for FrankeFunction with noise
        fig, ax = plt.subplots()
        plt.title("Bias-variance tradeoff train and test data w/noise\n "+ str(method) )
        fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
        ax.plot(complexity, p_e_t_noise, label="Train", color="g")
        ax.plot(complexity, p_e_noise, linestyle="--", label="Test", color="r")
        ax.set_xlabel("Model Complexity [polynomial degree]")
        ax.set_xticks(complexity[::2])
        ax.set_ylabel("Prediction Error")

        ax.text(0.1, 0.75, "High bias\nLow variance\n<------",
                horizontalalignment="left",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.text(0.9, 0.75, "Low bias\nHigh variance\n------>",
                horizontalalignment="right",
                verticalalignment="baseline",
                transform=ax.transAxes)
    
        ax.legend(loc=9)
        fig.tight_layout()
        fig.savefig(path+"biasvariancetradeoff_"+str(method)+"_Franke_noise.png", dpi=1000)
    
        # Plot for Variance of test data
        plt.figure()
        plt.title("Variance of the test data with "+str(method))
        plt.plot(complexity, error_var)
        plt.xlabel("Model Complexity [polynomial degree]")
        plt.savefig(path+"biasvar_Var_"+str(method)+".png", dpi=1000)
    
        # Plot for Bias of test data
        plt.figure()
        plt.title("Bias of the test data with "+str(method))
        plt.plot(complexity, error_bias)
        plt.xlabel("Model Complexity [polynomial degree]")
        plt.savefig(path+"biasvar_Bias_"+str(method)+".png", dpi=1000)
        plt.show()

    elif method == "Ridge":
        lambda_Ridge = np.logspace(-7, 1, 70)
        pred_error_ridge = np.zeros_like(lambda_Ridge)
        pred_error_train_ridge = np.zeros_like(pred_error_ridge)
        p_e_noise = np.zeros_like(pred_error_ridge)
        p_e_t_noise = np.zeros_like(pred_error_ridge)  
        error_var = np.zeros_like(pred_error_ridge)
        error_bias = np.zeros_like(pred_error_ridge)

        for j, lamb in enumerate(lambda_Ridge):
            if j % 10 == 0:
                print(j)
            pred_error_ridge[j], pred_error_train_ridge[j] = k_fold_CV_franke(x, y, Z, folds=5, 
                                                  dim=5, hyperparam=lamb, method="Ridge", train=True)
            p_e_noise[j], p_e_t_noise[j] = k_fold_CV_franke(x, y, Zn, folds=5, 
                                             dim=5, hyperparam=lamb, method="Ridge", train=True)
            # Calculate variance and Bias for test data
            _, _, error_var[j], error_bias[j] = k_fold_CV_franke(x, y, Zn, folds=5, 
                                                  dim=5, hyperparam=lamb, 
                                                  method="Ridge", train=False)

        # Plot for FrankeFunction
        pred_log = np.log10(pred_error_ridge)
        pred_log_train = np.log10(pred_error_train_ridge)
        fig, ax = plt.subplots()
        plt.title("Bias-variance tradeoff train and test data\n "+str(method))
        fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
        ax.plot(np.log10(lambda_Ridge), pred_log_train, label="Train", color="g")
        ax.plot(np.log10(lambda_Ridge), pred_log, linestyle="--", label="Test", color="r")
        ax.set_xlabel(r"log$_{10}\lambda$")
        ax.set_ylabel(r"log$_{10}$(Prediction Error)")
        ax.set_ylim([np.min(pred_log_train) - np.min(np.abs(pred_log_train)) * 0.1,
                     np.max(pred_log) + np.max(np.abs(pred_log)) * 0.3])
    
        ax.text(0.05, 0.75, "Low bias\nHigh variance\n<------",
                horizontalalignment="left",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.text(0.95, 0.75, "High bias\nLow variance\n------>",
                horizontalalignment="right",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.legend(loc=9)
        fig.tight_layout()
        fig.savefig(path+"biasvariancetradeoff_"+str(method)+"_Franke.png", dpi=1000)
        
        # Plot for FrankeFunction with noise
        pred_log = np.log10(p_e_noise)
        pred_log_train = np.log10(p_e_t_noise)
        fig, ax = plt.subplots()
        plt.title("Bias-variance tradeoff train and test data w/noise\n "+ str(method) )
        fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
        ax.plot(np.log10(lambda_Ridge), pred_log_train, label="Train", color="g")
        ax.plot(np.log10(lambda_Ridge), pred_log, linestyle="--", label="Test", color="r")
        ax.set_xlabel(r"log$_{10}\lambda$")
        ax.set_ylabel(r"log$_{10}$(Prediction Error)")
        ax.set_ylim([np.min(pred_log_train) - np.min(np.abs(pred_log_train)) * 0.1,
                 np.max(pred_log) + np.max(np.abs(pred_log)) * 0.3])

        ax.text(0.05, 0.75, "Low bias\nHigh variance\n<------",
                horizontalalignment="left",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.text(0.9, 0.75, "High bias\nLow variance\n------>",
                horizontalalignment="right",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.legend(loc=9)
        fig.tight_layout()
        fig.savefig(path+"biasvariancetradeoff_"+str(method)+"_Franke_noise.png", dpi=1000)
    
        # Plot for Variance of test data
        plt.figure()
        plt.title("Variance of the test data with "+str(method))
        plt.plot(np.log10(lambda_Ridge), error_var)
        plt.xlabel(r"log$_{10}\lambda$")
        plt.savefig(path+"biasvar_Var_"+str(method)+".png", dpi=1000)
    
        # Plot for Bias of test data
        plt.figure()
        plt.title("Bias of the test data with "+str(method))
        plt.plot(np.log10(lambda_Ridge), error_bias)
        plt.xlabel(r"log$_{10}\lambda$")
        plt.savefig(path+"biasvar_Bias_"+str(method)+".png", dpi=1000)
        plt.show()
    
    elif method == "Lasso":
        lambda_Lasso = np.logspace(-10, -1, 70)
        pred_error = np.zeros_like(lambda_Lasso)
        pred_error_train = np.zeros_like(pred_error)
        p_e_noise = np.zeros_like(pred_error)
        p_e_t_noise = np.zeros_like(pred_error)  
        error_var = np.zeros_like(pred_error)
        error_bias = np.zeros_like(pred_error)

        for j, lamb in enumerate(lambda_Lasso):
            if j % 10 == 0:
                print(j)
            pred_error[j], pred_error_train[j] = k_fold_CV_franke(x, y, Z, folds=5, 
                                                  dim=5, hyperparam=lamb, 
                                                  method="Lasso", train=True)
            p_e_noise[j], p_e_t_noise[j] = k_fold_CV_franke(x, y, Zn, folds=5, 
                                             dim=5, hyperparam=lamb, 
                                             method="Lasso", train=True)
            # Calculate variance and Bias for test data
            _, _, error_var[j], error_bias[j] = k_fold_CV_franke(x, y, Zn, folds=5, 
                                                   dim=5, hyperparam=lamb, 
                                                   method="Lasso", train=False)

        # Plot for FrankeFunction
        pred_log = np.log10(pred_error)
        pred_log_train = np.log10(pred_error_train)
        fig, ax = plt.subplots()
        plt.title("Bias-variance tradeoff train and test data\n "+str(method))
        fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
        ax.plot(np.log10(lambda_Lasso), pred_log_train, label="Train", color="g")
        ax.plot(np.log10(lambda_Lasso), pred_log, linestyle="--", label="Test", color="r")
        ax.set_xlabel(r"log$_{10}\lambda$")
        ax.set_ylabel(r"log$_{10}$(Prediction Error)")
        ax.set_ylim([np.min(pred_log_train) - np.min(np.abs(pred_log_train)) * 0.1,
                     np.max(pred_log) + np.max(np.abs(pred_log)) * 0.3])
    
        ax.text(0.05, 0.75, "Low bias\nHigh variance\n<------",
                horizontalalignment="left",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.text(0.95, 0.75, "High bias\nLow variance\n------>",
                horizontalalignment="right",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.legend(loc=9)
        fig.tight_layout()
        fig.savefig(path+"biasvariancetradeoff_"+str(method)+"_Franke.png", dpi=1000)
        
        # Plot for FrankeFunction with noise
        pred_log = np.log10(p_e_noise)
        pred_log_train = np.log10(p_e_t_noise)
        fig, ax = plt.subplots()
        plt.title("Bias-variance tradeoff train and test data w/noise\n "+ str(method) )
        fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
        ax.plot(np.log10(lambda_Lasso), pred_log_train, label="Train", color="g")
        ax.plot(np.log10(lambda_Lasso), pred_log, linestyle="--", label="Test", color="r")
        ax.set_xlabel(r"log$_{10}\lambda$")
        ax.set_ylabel(r"log$_{10}$(Prediction Error)")
        ax.set_ylim([np.min(pred_log_train) - np.min(np.abs(pred_log_train)) * 0.1,
                 np.max(pred_log) + np.max(np.abs(pred_log)) * 0.3])

        ax.text(0.05, 0.75, "Low bias\nHigh variance\n<------",
                horizontalalignment="left",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.text(0.9, 0.75, "High bias\nLow variance\n------>",
                horizontalalignment="right",
                verticalalignment="baseline",
                transform=ax.transAxes)
    
        ax.legend(loc=9)
        fig.tight_layout()
        fig.savefig(path+"biasvariancetradeoff_"+str(method)+"_Franke_noise.png", dpi=1000)
    
        # Plot for Variance of test data
        plt.figure()
        plt.title("Variance of the test data with "+str(method))
        plt.plot(np.log10(lambda_Lasso), error_var)
        plt.xlabel(r"log$_{10}\lambda$")
        fig.tight_layout()
        plt.savefig("figures/biasvar_Var_"+str(method)+".png", dpi=1000)
    
        # Plot for Bias of test data
        plt.figure()
        plt.title("Bias of the test data with "+str(method))
        plt.plot(np.log10(lambda_Lasso), error_bias)
        plt.xlabel(r"log$_{10}\lambda$")
        fig.tight_layout()
        plt.savefig(path+"biasvar_Bias_"+str(method)+".png", dpi=1000)
        plt.show()

def seaborn_heatmap(R2, lambda_params, dim_params, noise="", method=""):
    """Make heatmap for the R2 score function for the methods"""
    sns.set()
    path = "figures/"
    if noise == "No":
        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(R2, vmin=-1, vmax=1, annot=True, ax=ax, cmap="viridis",
                xticklabels=lambda_params, yticklabels=dim_params, fmt=".3f")
        ax.set_title("R2-score")
        ax.set_ylabel("Model complexity")
        ax.set_xlabel("$\lambda$")
        fig.savefig(path+"R2_score_without_noise_heatmap_"+str(method)+".png")

    elif noise == "Yes":
        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(R2, vmin=-1, vmax=1, annot=True, ax=ax, cmap="viridis",
                xticklabels=lambda_params, yticklabels=dim_params, fmt=".3f")
        ax.set_title("R2-score with noise")
        ax.set_ylabel("Model complexity")
        ax.set_xlabel("$\lambda$")
        fig.savefig(path+"R2_score_with_noise_heatmap_"+str(method)+".png")

    elif noise == None:
        path = "terrain_figures/"
        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(R2, vmin=-1, vmax=1, annot=True, ax=ax, cmap="viridis",
                xticklabels=lambda_params, yticklabels=dim_params, fmt=".3f")
        ax.set_title("R2-score for terrain data")
        ax.set_ylabel("Model complexity")
        ax.set_xlabel("$\lambda$")
        fig.savefig(path+"R2_score_terrain_heatmap_"+str(method)+".png")
    
    plt.show()

@ignore_warnings(category = ConvergenceWarning)
def terrain_plot(x, y, z, p, plots=""):
    from funcs import k_fold_CV_terrain
    path = "terrain_figures/"
    """
    Function that loads the terrain data of Hardangervidda, and shows it.
    """
    if plots == "terrain":
        # Load the terrain
        vidda = imread("n59_e010_1arc_v3.tif")
        #print(terrain1.shape)
        # Show the terrain
        plt.figure()
        plt.title("Terrain over Hardangervidda")
        plt.imshow(vidda, cmap="gray")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(path+"2d_terrain.png")
        
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
    
        # Plot the surface.
        surf = ax.plot_surface(x_grid, y_grid, vidda, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha = 0.5)
        ax.view_init(azim=30)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.savefig(path+"3d_terrain.png")
        plt.show()
    
    elif plots == "OLS":
        """
        Function to plot the test and training errors as
        functions of model complexity (p).
        """
        complexity = np.arange(0, p+1)
        pred_error = np.zeros_like(complexity, dtype=float)        
        error_var = np.zeros_like(pred_error)
        error_bias = np.zeros_like(pred_error)    
        pred_error_train = np.zeros_like(pred_error)
        
        for j in complexity:
            print(j)
            pred_error[j], pred_error_train[j] = k_fold_CV_terrain(x, y, z, folds=5, 
                                                  dim=j, hyperparam=1, 
                                                  method="OLS", Train=True)
            # Calculate variance and Bias for test data
            _, _, error_var[j], error_bias[j] = k_fold_CV_terrain(x, y, z, folds=5, 
                                                   dim=j, hyperparam=1, 
                                                   method="OLS", Train=False)
        
        # Plot for FrankeFunction
        #pred_log = np.log10(pred_error)
        #pred_log_train = np.log10(pred_error_train)
        fig, ax = plt.subplots()
        plt.title("Bias-variance tradeoff train and test data\n "+str(plots)+" for terrain data")
        fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
        ax.plot(complexity, pred_error_train, label="Train", color="g")
        ax.plot(complexity, pred_error, linestyle="--", label="Test", color="r")
        ax.set_xlabel("Model Complexity [polynomial degree]")
        ax.set_xticks(complexity[::2])
        ax.set_ylabel("Prediction Error")#(r"log$_{10}$(Prediction Error)")
        #ax.set_ylim([np.min(pred_log_train) - np.min(np.abs(pred_log_train)) * 0.1,
                 #np.max(pred_log) + np.max(np.abs(pred_log)) * 0.3])

        ax.text(0.1, 0.75, "High bias\nLow variance\n<------",
                horizontalalignment="left",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.text(0.95, 0.75, "Low bias\nHigh variance\n------>",
                horizontalalignment="right",
                verticalalignment="baseline",
                transform=ax.transAxes)

        ax.legend(loc=9)
        fig.tight_layout()
        fig.savefig(path+"biasvariancetradeoff_"+str(plots)+"_terrain.png", dpi=1000)
    
        # Plot for Variance of test data
        plt.figure()
        plt.title("Variance of the test data with "+str(plots)+"\n for terrain data")
        plt.plot(complexity, error_var)
        plt.xlabel("Model Complexity [polynomial degree]")
        plt.savefig(path+"biasvar_Var_"+str(plots)+"_terrain.png", dpi=1000)
    
        # Plot for Bias of test data
        plt.figure()
        plt.title("Bias of the test data with "+str(plots)+"\n for terrain data")
        plt.plot(complexity, error_bias)
        plt.xlabel("Model Complexity [polynomial degree]")
        plt.savefig(path+"biasvar_Bias_"+str(plots)+"_terrain.png", dpi=1000)
        plt.show()

    elif plots == "Ridge":
        lambda_Ridge = np.logspace(-7, 1, 70)
        pred_error_ridge = np.zeros_like(lambda_Ridge)
        pred_error_train_ridge = np.zeros_like(pred_error_ridge)
        error_var = np.zeros_like(pred_error_ridge)
        error_bias = np.zeros_like(pred_error_ridge)

        for j, lamb in enumerate(lambda_Ridge):
            if j % 10 == 0:
                print(j)
            pred_error_ridge[j], pred_error_train_ridge[j] = k_fold_CV_terrain(x, y, z, folds=5, 
                                                  dim=5, hyperparam=lamb, method="Ridge", Train=True)
            # Calculate variance and Bias for test data
            _, _, error_var[j], error_bias[j] = k_fold_CV_terrain(x, y, z, folds=5, 
                                                  dim=5, hyperparam=lamb, 
                                                  method="Ridge", Train=False)

        # Plot for FrankeFunction
        pred_log = np.log10(pred_error_ridge)
        pred_log_train = np.log10(pred_error_train_ridge)
        fig, ax = plt.subplots()
        plt.title("Bias-variance tradeoff train and test data\n "+str(plots)+" for terrain data")
        fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
        ax.plot(np.log10(lambda_Ridge), pred_log_train, label="Train", color="g")
        ax.plot(np.log10(lambda_Ridge), pred_log, linestyle="--", label="Test", color="r")
        ax.set_xlabel(r"log$_{10}\lambda$")
        ax.set_ylabel(r"log$_{10}$(Prediction Error)")
        ax.set_ylim([np.min(pred_log_train) - np.min(np.abs(pred_log_train)) * 0.1,
                     np.max(pred_log) + np.max(np.abs(pred_log)) * 0.3])
    
        ax.text(0.05, 0.75, "Low bias\nHigh variance\n<------",
                horizontalalignment="left",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.text(0.95, 0.75, "High bias\nLow variance\n------>",
                horizontalalignment="right",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.legend(loc=9)
        fig.tight_layout()
        fig.savefig(path+"biasvariancetradeoff_"+str(plots)+"_terrain.png", dpi=1000)
    
        # Plot for Variance of test data
        plt.figure()
        plt.title("Variance of the test data with "+str(plots)+"\n for terrain data")
        plt.plot(np.log10(lambda_Ridge), error_var)
        plt.xlabel(r"log$_{10}\lambda$")
        plt.savefig(path+"biasvar_Var_"+str(plots)+"_terrain.png", dpi=1000)
    
        # Plot for Bias of test data
        plt.figure()
        plt.title("Bias of the test data with "+str(plots)+"\n for terrain data")
        plt.plot(np.log10(lambda_Ridge), error_bias)
        plt.xlabel(r"log$_{10}\lambda$")
        plt.savefig(path+"biasvar_Bias_"+str(plots)+"_terrain.png", dpi=1000)
        plt.show()
    
    elif plots == "Lasso":
        lambda_Lasso = np.logspace(-10, -1, 70)
        pred_error = np.zeros_like(lambda_Lasso)
        pred_error_train = np.zeros_like(pred_error)
        error_var = np.zeros_like(pred_error)
        error_bias = np.zeros_like(pred_error)

        for j, lamb in enumerate(lambda_Lasso):
            if j % 10 == 0:
                print(j)
            pred_error[j], pred_error_train[j] = k_fold_CV_terrain(x, y, z, folds=5, 
                                                  dim=5, hyperparam=lamb, 
                                                  method="Lasso", Train=True)
            # Calculate variance and Bias for test data
            _, _, error_var[j], error_bias[j] = k_fold_CV_terrain(x, y, z, folds=5, 
                                                   dim=5, hyperparam=lamb, 
                                                   method="Lasso", Train=False)

        # Plot for FrankeFunction
        pred_log = np.log10(pred_error)
        pred_log_train = np.log10(pred_error_train)
        fig, ax = plt.subplots()
        plt.title("Bias-variance tradeoff train and test data\n "+str(plots)+" for terrain data")
        fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
        ax.plot(np.log10(lambda_Lasso), pred_log_train, label="Train", color="g")
        ax.plot(np.log10(lambda_Lasso), pred_log, linestyle="--", label="Test", color="r")
        ax.set_xlabel(r"log$_{10}\lambda$")
        ax.set_ylabel(r"log$_{10}$(Prediction Error)")
        ax.set_ylim([np.min(pred_log_train) - np.min(np.abs(pred_log_train)) * 0.1,
                     np.max(pred_log) + np.max(np.abs(pred_log)) * 0.3])
    
        ax.text(0.05, 0.75, "Low bias\nHigh variance\n<------",
                horizontalalignment="left",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.text(0.95, 0.75, "High bias\nLow variance\n------>",
                horizontalalignment="right",
                verticalalignment="baseline",
                transform=ax.transAxes)
        ax.legend(loc=9)
        fig.tight_layout()
        fig.savefig(path+"biasvariancetradeoff_"+str(plots)+"_terrain.png", dpi=1000)
    
        # Plot for Variance of test data
        plt.figure()
        plt.title("Variance of the test data with "+str(plots)+"\n for terrain data")
        plt.plot(np.log10(lambda_Lasso), error_var)
        plt.xlabel(r"log$_{10}\lambda$")
        fig.tight_layout()
        plt.savefig(path+"biasvar_Var_"+str(plots)+"_terrain.png", dpi=1000)
    
        # Plot for Bias of test data
        plt.figure()
        plt.title("Bias of the test data with "+str(plots)+"\n for terrain data")
        plt.plot(np.log10(lambda_Lasso), error_bias)
        plt.xlabel(r"log$_{10}\lambda$")
        fig.tight_layout()
        plt.savefig(path+"biasvar_Bias_"+str(plots)+"_terrain.png", dpi=1000)
        plt.show()
