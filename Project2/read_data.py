import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer




# Reading file into data frame
cwd = os.getcwd()
filename = cwd + "/default of credit card clients.xls"
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0)

df.rename(index=str, columns={"default payment next month": "dpnm"}, inplace=True)

# Features/design matrix X and targets y
X = df.loc[:, (df.columns != "dpnm")].values
y = df["dpnm"].values

# Remove feature outliners
# X2: Sex (1 = male; 2 = female)
y = y[np.logical_and(X[:, 1] >= 1, X[:, 1] <= 2)]
X = X[np.logical_and(X[:, 1] >= 1, X[:, 1] <= 2)]

# X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)
y = y[np.logical_and(X[:, 2] >= 1, X[:, 2] <= 4)]
X = X[np.logical_and(X[:, 2] >= 1, X[:, 2] <= 4)]

# X4: Marital status (1 = married; 2 = single; 3 = others)
y = y[np.logical_and(X[:, 3] >= 1, X[:, 3] <= 3)]
X = X[np.logical_and(X[:, 3] >= 1, X[:, 3] <= 3)]

feature = df.loc[:, (df.columns != "dpnm")]
X_ = pd.DataFrame(X, columns=feature.columns.values)

col_encode = list(X_.columns.values[1:4])
col_scale = [X_.columns.values[0]] + list(X_.columns.values[4:])

encoder = OneHotEncoder(categories="auto", sparse=False)
scaler = StandardScaler()

encoded_cols = encoder.fit_transform(X_[col_encode])
scaled_cols = scaler.fit_transform(X_[col_scale])

X_trim = np.concatenate([encoded_cols, scaled_cols], axis=1)

X_fin = np.append(np.ones_like(X_trim[:, 1]).reshape(-1, 1), X_trim, axis=1)
y_fin = y

np.save("Data/matrix_credit.npy", X_fin)
np.save("Data/targets_credit.npy", y_fin)
