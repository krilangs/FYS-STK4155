import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
import scipy.integrate as scint

from main import LogReg


fonts = {"font.family": "serif", "axes.labelsize": 8, "font.size": 8,
         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
         "axes.titlesize": 8}

plt.rcParams.update(fonts)

test_set = np.load("Data/Credit_test.npz")
train_set = np.load("Data/Credit_train.npz")
X_test, y_test = test_set["X_test"], test_set["y_test"].reshape(-1, 1)
X_train, y_train = train_set["X_train"], train_set["y_train"].reshape(-1, 1)

model = LogReg()
model.load_model("LR_credit_model.npz")

y_pred = model.predict_prob(X_test)
prob0 = 1 - y_pred
prob1 = y_pred

prob_split = np.append(prob0, prob1, axis=1)


def bestCurve(y):
    defaults = np.sum(y == 1, dtype=int)
    total = len(y)
    x = np.linspace(0, 1, total, endpoint=True)
    y1 = np.linspace(0, 1, defaults, endpoint=True)
    y2 = np.ones(total - defaults)
    y3 = np.concatenate([y1, y2])
    return x, y3


x, gains_best = bestCurve(y_test)
fig, ax = plt.subplots()
skplt.metrics.plot_cumulative_gain(y_test.ravel(), prob_split, ax=ax)
plt.title("Lift chart of Logistic Regression \n for Classification")
ax.plot(x, gains_best)
ax.legend(["Not default", "Default", "Baseline", "Best model"])
ax.axis([x[0], x[-1], 0, 1.01])
fig.set_size_inches(3.03, 3.03)
fig.tight_layout()
fig.savefig("Figures/cumulative_gain_logreg.png", dpi=1000)
fig.clf()


area_baseline = 0.5
area_best = scint.simps(gains_best, x) - area_baseline

x, gains_0 = skplt.helpers.cumulative_gain_curve(y_test.ravel(), prob_split[:,0], 0)
area_0 = scint.simps(gains_0, x) - area_baseline

x, gains_1 = skplt.helpers.cumulative_gain_curve(y_test.ravel(), prob_split[:,1], 1)
area_1 = scint.simps(gains_1, x) - area_baseline


ratio_not_default = area_0 / area_best
ratio_default = area_1 / area_best

print(f"Area ratio for predicting not default: {ratio_not_default:.6f}\n"
      + f"Area ratio for predicting default: {ratio_default:.6f} \n"
      + f"Error rate test: {1 - model.accuracy_score(X_test, y_test):.6f} \n"
      + f"Error rate train: {1 - model.accuracy_score(X_train, y_train):.6f} \n"
      + f"Accuracy score test: {model.accuracy_score(X_test, y_test):.6f} \n"
      + f"Accuracy score train: {model.accuracy_score(X_train, y_train):.6f}")


df = pd.read_csv("Data/train_credit_LR.csv", header=None, skiprows=1).T

df.columns = df.iloc[0]
df.drop(0, inplace=True)
df["param_learning_rate"] = pd.to_numeric(df["param_learning_rate"])
df = df.sort_values(by="param_learning_rate", ascending=True)

train_score = df["mean_train_score"].values.astype(np.float)
validation_score = df["mean_test_score"].values.astype(np.float)
learning_rates = df["param_learning_rate"].values.astype(np.float)
best_learning_rate = learning_rates[validation_score == np.max(validation_score)][0]

print(f"Best learning rate: {best_learning_rate:e}")

fig, ax = plt.subplots()
fig.set_size_inches(3.03, 3.03)
plt.title("Learning rate accuracy:\n Logistic Regression")
ax.semilogx(learning_rates, train_score, label="train")
ax.semilogx(learning_rates, validation_score, label="validation")
ax.legend()
ax.grid()
ax.set_xlabel(r"Learning rate $\gamma$")
ax.set_ylabel("Accuracy")
fig.tight_layout()
fig.savefig("Figures/logreg_learning_rate_accuracy.png", dpi=1000)
fig.clf()
