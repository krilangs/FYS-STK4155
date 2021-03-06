import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scikitplot as skplt
import scipy.integrate as scint

from main import MLPClassifier

# Set font sizes for plotting
fonts = {"font.family": "serif", "axes.labelsize": 8, "font.size": 8,
         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
         "axes.titlesize": 8}

plt.rcParams.update(fonts)

# Import the data
test_set = np.load("Data/credit_test.npz")
train_set = np.load("Data/credit_train.npz")
X_test, y_test = test_set["X_test"], test_set["y_test"].reshape(-1, 1)
X_train, y_train = train_set["X_train"], train_set["y_train"].reshape(-1, 1)

model = MLPClassifier()
model.load_model("NN_credit_model.npz")
y_pred = model.predict_prob(X_test)
prob0 = 1 - y_pred
prob1 = y_pred

prob_split = np.append(prob0, prob1, axis=1)

def bestCurve(y):
    defaults = np.sum(y == 1, dtype=np.int)
    total = len(y)
    x = np.linspace(0, 1, total, endpoint=True)
    y1 = np.linspace(0, 1, defaults, endpoint=True)
    y2 = np.ones(total - defaults)
    y3 = np.concatenate([y1, y2])
    return x, y3

x, gains_best = bestCurve(y_test)
fig, ax = plt.subplots()
skplt.metrics.plot_cumulative_gain(y_test.ravel(), prob_split, ax=ax)
plt.title("Lift chart of Neural Network\n for Classification")
ax.plot(x, gains_best)
ax.legend(["Not default", "Default", "Baseline", "Best model"])
ax.axis([x[0], x[-1], 0, 1.01])
fig.set_size_inches(3.03, 3.03)
fig.tight_layout()
fig.savefig("Figures/cumulative_gain_NN.png", dpi=1000)
fig.clf()

area_baseline = 0.5
area_best = scint.simps(gains_best, x) - area_baseline

x, gains_0 = skplt.helpers.cumulative_gain_curve(y_test.ravel(), prob_split[:,0], 0)
area_0 = scint.simps(gains_0, x) - area_baseline

x, gains_1 = skplt.helpers.cumulative_gain_curve(y_test.ravel(), prob_split[:,1], 1)
area_1 = scint.simps(gains_1, x) - area_baseline


ratio_not_default = area_0/area_best
ratio_default = area_1/area_best

df = pd.read_csv("Data/train_credit_NN.csv", header=None, skiprows=1).T

df.columns = df.iloc[0]
df.drop(0, inplace=True)
df["rank_test_score"] = pd.to_numeric(df["rank_test_score"])
df = df.sort_values(by="param_learning_rate", ascending=True)

train_score = df["mean_train_score"].values.astype(np.float)
validation_score = df["mean_test_score"].values.astype(np.float)
learning_rates = df["param_learning_rate"].values.astype(np.float)
lambdas = df["param_lambd"].values.astype(np.float)
best_learning_rate = learning_rates[validation_score == np.max(validation_score)][0]
best_lambda = lambdas[validation_score == np.max(validation_score)][0]

print(f"Area ratio for predicting not default: {ratio_not_default:.6f}\n"
      + f"Area ratio for predicting default: {ratio_default:.6f}\n"
      + f"Error rate test: {1 - model.accuracy_score(X_test, y_test):.6f}\n"
      + f"Error rate train: {1 - model.accuracy_score(X_train, y_train):.6f}\n"
      + f"Accuracy score test: {model.accuracy_score(X_test, y_test):.6f}\n"
      + f"Accuracy score train: {model.accuracy_score(X_train, y_train):.6f}\n"
      + f"Best lambda: {best_lambda:e}\n"
      + f"Best learning rate: {best_learning_rate:e}")

fig, ax = plt.subplots()
fig.set_size_inches(3.03, 3.03)
plt.title("Learning rate and regularization\n accuracy:\n Neural Network - Classification")
ax.scatter(learning_rates, lambdas, c=validation_score, s=20, cmap=cm.coolwarm)
ax.set_xlabel(r"Learning rate $\gamma$")
ax.set_ylabel(r"Regularization parameter $\lambda$")
ax.set_xlim([np.min(learning_rates)*0.9, np.max(learning_rates)*1.1])
ax.set_ylim([np.min(lambdas)*0.9, np.max(lambdas)*1.1])
ax.set_yscale("log")
ax.set_xscale("log")
cbar = fig.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(
        vmin=validation_score.min(), vmax=validation_score.max()),
        cmap=cm.coolwarm), ax=ax)
cbar.set_label("Validation accuracy")
fig.tight_layout()
fig.savefig("Figures/nn_learning_rate_lambda_accuracy_credit.png", dpi=1000)
fig.clf()
