"""
File containing functions that create plots,
and which are imported to the main file.
"""
from mpl_toolkits.mplot3d import Axes3D
from imageio import imread
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot3d(x, y, z, z2):
    """
    Function to make a 3d plot of the Franke function by using OLS.
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    #ax = fig.add_subplot(121, projection = "3d")
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig = plt.figure()
    #ax = fig.add_subplot(122, projection = "3d")
    ax = fig.gca(projection="3d")
    # Plot the surface.
    surf = ax.plot_surface(x, y, z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def terrain():
    """
    Function that loads the terrain data of Hardangervidda, and shows it.
    """
    # Load the terrain
    terrain1 = imread("n59_e010_1arc_v3.tif")
    #print(terrain1.shape)
    # Show the terrain
    plt.figure()
    plt.title("Terrain over Hardangervidda")
    plt.imshow(terrain1, cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
