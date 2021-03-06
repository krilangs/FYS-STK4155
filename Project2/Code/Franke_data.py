import sys
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(777)

def Generate_data(nx, ny, std=0):
    """
    Use Franke's function to ganerate data set with:
    nx: Number of points in the x-direction, int
    ny: Number of points in the y-direction, int
    std: Standard deviation of the added Gaussian noise
    """
    x = np.linspace(0, 1, nx, endpoint=True)
    y = np.linspace(0, 1, ny, endpoint=True)

    X, Y = np.meshgrid(x, y)

    term1 = 0.75*np.exp(-(0.25*(9*X - 2)**2) - 0.25*((9*Y - 2)**2))
    term2 = 0.75*np.exp(-((9*X + 1)**2)/49.0 - 0.1*(9*Y + 1))
    term3 = 0.5*np.exp(-(9*X - 7)**2/4.0 - 0.25*((9*Y - 3)**2))
    term4 = -0.2*np.exp(-(9*X - 4)**2 - (9*Y - 7)**2)

    Z = term1 + term2 + term3 + term4 + std*np.random.normal(0, 1, size=term1.shape)
    return (X, Y, Z)

try:
    nx = int(sys.argv[1])
    ny = int(sys.argv[2])
    sigma = float(sys.argv[3])
except IndexError:
    raise IndexError(f"Input the number of points in x direction, y direction"
                     + f" and the standard deviation")
except ValueError:
    raise TypeError("Input must be: int, int, float")

x, y, z = Generate_data(nx, ny, sigma)

x_meshgrid = x.copy()
y_meshgrid = y.copy()
z_meshgrid = z.copy()

x = x.ravel()
y = y.ravel()
z = z.ravel().reshape(-1, 1)

X = np.array([x, y]).T

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.28)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Export the Franke function data
joblib.dump(scaler, f"Models/Scaler_data_features_{nx}_{ny}_{sigma}.pkl")
np.savez(f"Data/Franke_train_{nx}_{ny}_{sigma}.npz", X_train=X_train, z_train=z_train)
np.savez(f"Data/Franke_test_{nx}_{ny}_{sigma}.npz", X_test=X_test, z_test=z_test)
np.savez(f"Data/Franke_meshgrid_{nx}_{ny}_{sigma}.npz", x=x_meshgrid, y=y_meshgrid, z=z_meshgrid)
