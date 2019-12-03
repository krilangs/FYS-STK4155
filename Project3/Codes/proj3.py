import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from IPython.display import Image
import os
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.externals.six import StringIO
import pydotplus

from IPython.display import display
# Read file into data frame, and inspect the data set
infile = open("pulsar_stars.csv", "r")

pulsar_data = pd.read_csv(infile, header=0,
                names = ("Mean_Int", "STD_Int", "Ex_Kurt_Int", "Skew_Int",
                         "Mean_DMSNR", "STD_DMSNR", "Ex_Kurt_DMSNR", "Skew_DMSNR",
                         "Target"))
#pulsar_data.info()
pulsar_data = pd.DataFrame(pulsar_data)
#print(pulsar_data.head())
#f, ax = plt.subplots(figsize=(15,15))
#sns.heatmap(pulsar_data.corr(), annot=True, linecolor="blue", fmt=".2f", ax=ax)

# Create design matrix and targets
X = pulsar_data.loc[:, pulsar_data.columns != "Target"].values
y = pulsar_data.loc[:, pulsar_data.columns == "Target"].values


#encoder = OneHotEncoder(handle_unknown="ignore")
#encoder.fit(X)
#X = encoder.transform(X)


tree_clf = DecisionTreeClassifier(max_depth=5)
tree_clf.fit(X, y)
"""
show = pulsar_data.loc[:, (pulsar_data.columns != "Target")]
print(list(show.columns))
export_graphviz(tree_clf, out_file="Data/tree.dot", filled=True, rounded=True, special_characters=True,
                proportion=False, class_names="Target", feature_names=list(show.columns))
cmd = 'dot -Tpng Data/tree.dot -o Data/tree.png'
os.system(cmd)
"""


print("Train set accuracy with Decision Tree: {:.6f}".format(tree_clf.score(X,y)))
print(cross_val_score(tree_clf, X, y, cv=5))


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
y = np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)
scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')
voting_clf.fit(X_train, y_train)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf, tree_clf, bag_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

