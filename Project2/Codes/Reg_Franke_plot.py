import sys
import numpy as np
import joblib
import matplotlib
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
from main import MultilayerPerceptronRegressor

try:
    nx = int(sys.argv[1])
    ny = int(sys.argv[2])
    sigma = float(sys.argv[3])
except IndexError:
    raise IndexError(
        f"Input the number of points in x direction, y direction"
        + f" and the standard deviation"
    )
except ValueError:
    raise TypeError("Input must be integer, integer and float")
#nx = 50
#ny = 50
#sigma = 0.2

fonts = {
    "font.family": "serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}

plt.rcParams.update(fonts)

meshgrid = np.load(f"Data/Franke_meshgrid_{nx}_{ny}_{sigma}.npz")

x_meshgrid, y_meshgrid, z_meshgrid = meshgrid["x"], meshgrid["y"], meshgrid["z"]


training_set = np.load(f"Data/Franke_train_{nx}_{ny}_{sigma}.npz")
test_set = np.load(f"Data/Franke_test_{nx}_{ny}_{sigma}.npz")

X_train, z_train = training_set["X_train"], training_set["z_train"].reshape(-1, 1)
X_test, z_test = test_set["X_test"], test_set["z_test"].reshape(-1, 1)


scaler = joblib.load(f"Models/Scaler_data_features_{nx}_{ny}_{sigma}.pkl")

model = MultilayerPerceptronRegressor()
model.load_model(f"Reg_Franke_model_{nx}_{ny}_{sigma}.npz")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


fig = plt.figure()
fig.set_size_inches(3.03, 3.03)
ax = fig.gca(projection="3d")

surf = ax.plot_surface(
    x_meshgrid,
    y_meshgrid,
    z_meshgrid,
    cmap=matplotlib.cm.coolwarm,
    linewidth=0,
    antialiased=False,
    alpha=0.5,
)

ax.scatter(
    scaler.inverse_transform(X_test)[:, 0],
    scaler.inverse_transform(X_test)[:, 1],
    y_pred_test,
    marker=".",
    s=7,
    label="test",
)

ax.axis("off")
ax.grid(False)
ax.set_frame_on(False)
fig.savefig(
    f"Figures/3dplot_test_{nx}_{ny}_{sigma}.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=1000,
)

fig = plt.figure()
fig.set_size_inches(3.03, 3.03)
ax = fig.gca(projection="3d")

surf = ax.plot_surface(
    x_meshgrid,
    y_meshgrid,
    z_meshgrid,
    cmap=matplotlib.cm.coolwarm,
    linewidth=0,
    antialiased=False,
    alpha=0.5,
)

ax.scatter(
    scaler.inverse_transform(X_train)[:, 0],
    scaler.inverse_transform(X_train)[:, 1],
    y_pred_train,
    marker=".",
    s=7,
    label="train",
)

ax.axis("off")
ax.grid(False)
ax.set_frame_on(False)
fig.savefig(
    f"Figures/3dplot_train_{nx}_{ny}_{sigma}.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=1000,
)

# Plotting hyperparameter search
df = pd.read_csv(
    f"Data/train_franke_NN_{nx}_{ny}_{sigma}.csv", header=None, skiprows=1
).T

df.columns = df.iloc[0]
df.drop(0, inplace=True)
df["rank_test_score"] = pd.to_numeric(df["rank_test_score"])
df = df.sort_values(by="param_learning_rate", ascending=True)

train_score = df["mean_train_score"].values.astype(np.float)
validation_score = df["mean_test_score"].values.astype(np.float)
# Making all strongly negative scores default to 1.
validation_score[validation_score < -1] = -1
learning_rates = df["param_learning_rate"].values.astype(np.float)
lambdas = df["param_lambd"].values.astype(np.float)
best_learning_rate = learning_rates[validation_score == np.max(validation_score)][0]
best_lambda = lambdas[validation_score == np.max(validation_score)][0]
fig, ax = plt.subplots()
fig.set_size_inches(3.03, 3.03)
ax.scatter(learning_rates, lambdas, c=validation_score, s=20, cmap=cm.coolwarm)
ax.set_xlabel(r"Learning rate $\eta$")
ax.set_ylabel(r"Shrinkage parameter $\lambda$")
ax.set_xlim([np.min(learning_rates) * 0.9, np.max(learning_rates) * 1.1])
ax.set_ylim([np.min(lambdas) * 0.9, np.max(lambdas) * 1.1])
ax.set_yscale("log")
ax.set_xscale("log")
cbar = fig.colorbar(
    cm.ScalarMappable(
        norm=cm.colors.Normalize(
            vmin=validation_score.min(), vmax=validation_score.max()
        ),
        cmap=cm.coolwarm,
    ),
    ax=ax,
)
cbar.set_label(r"R$^2$ score")
fig.tight_layout()
fig.savefig(
    f"Figures/nn_learning_rate_lambda_r2_franke_{nx}_{ny}_{sigma}.png",
    dpi=1000,
)
fig.clf()


print(f"R2 score test: {model.r2_score(X_test, z_test)}.\n"
      + f"R2 score train: {model.r2_score(X_train, z_train)}")
print(f"Best lambda: {best_lambda:g}\n"
      + f"Best learning rate: {best_learning_rate:e}")