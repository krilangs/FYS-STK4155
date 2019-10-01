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
