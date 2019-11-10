import os
import imblearn
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Reading file into data frame
cwd = os.getcwd()
filename = cwd + "/Data/default of credit card clients.xls"
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0)
df.rename(index=str, columns={"default payment next month": "dpnm"}, inplace=True)

# Remove instances with zeros only for past bill statements or paid amounts
df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0) &
                (df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)

# Features/design matrix X and targets y
feature = df.loc[:, (df.columns != "dpnm")]
X = feature.values
y = df["dpnm"].values

# Remove feature outliners
# X2: Sex, should have values (1, 2)
y = y[np.logical_and(X[:, 1] >= 1, X[:, 1] <= 2)]
X = X[np.logical_and(X[:, 1] >= 1, X[:, 1] <= 2)]

# X3: Education, should have values (1, 2, 3, 4,)
y = y[np.logical_and(X[:, 2] >= 1, X[:, 2] <= 4)]
X = X[np.logical_and(X[:, 2] >= 1, X[:, 2] <= 4)]

# X4: Marital status, should have values (1, 2, 3)
y = y[np.logical_and(X[:, 3] >= 1, X[:, 3] <= 3)]
X = X[np.logical_and(X[:, 3] >= 1, X[:, 3] <= 3)]

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, stratify=y)

# Oversampling to get equal ratio of targets 0 and 1
res = imblearn.over_sampling.RandomOverSampler(sampling_strategy=1)
X_train, y_train = res.fit_resample(X_train, y_train)

# Make dataframes of scaled features
X_tr = pd.DataFrame(X_train, columns=feature.columns.values)
X_te = pd.DataFrame(X_test, columns=feature.columns.values)

# Onehot-encode categotical features
col_encode = list(X_tr.columns.values[1:4])

encoder = OneHotEncoder(categories="auto", sparse=False)

encoded_cols_train = encoder.fit_transform(X_tr[col_encode])
encoded_cols_test = encoder.fit_transform(X_te[col_encode])

# Scale the rest of the features
col_scale = [X_tr.columns.values[0]] + list(X_tr.columns.values[4:])

scaler = StandardScaler()
scaled_cols = scaler.fit(X_tr[col_scale])
scale_train = scaled_cols.transform(X_tr[col_scale])
scale_test = scaled_cols.transform(X_te[col_scale])

# Combine into one array for training and one for testing
X_trim_train = np.concatenate([encoded_cols_train, scale_train], axis=1)
X_trim_test = np.concatenate([encoded_cols_test, scale_test], axis=1)

# Export the preprocessed data
np.savez("Data/Credit_train.npz", X_train=X_trim_train, y_train=y_train)
np.savez("Data/Credit_test.npz", X_test=X_trim_test, y_test=y_test)
