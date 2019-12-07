import os
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import scikitplot as skplt
import sklearn.metrics as sklm
import sklearn.ensemble as skle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score, train_test_split


# Read file into data frame, and inspect the data set
infile = open("pulsar_stars.csv", "r")

pulsar_data = pd.read_csv(infile, header=0,
                          names=("Mean_Int", "STD_Int", "Ex_Kurt_Int", "Skew_Int",
                         "Mean_DMSNR", "STD_DMSNR", "Ex_Kurt_DMSNR", "Skew_DMSNR",
                         "Target"))
pulsar_data = pd.DataFrame(pulsar_data)

# Create design matrix and targets
feature = pulsar_data.loc[:, pulsar_data.columns != "Target"]
X = feature.values
y = pulsar_data.loc[:, pulsar_data.columns == "Target"].values



# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=0.27,
                                                    random_state=42)

# Scale training and test design matrix data
scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

# Information gain of the features
cols = list(feature.columns)
infos = mutual_info_classif(X_train, y_train, random_state=42)
info_gain_int = {cols[0]:[infos[0]], cols[1]:[infos[1]],
                 cols[2]:[infos[2]], cols[3]:[infos[3]]}
info_gain_int = pd.DataFrame(info_gain_int)
info_gain_curve = {cols[4]:[infos[4]], cols[5]:[infos[5]],
                   cols[6]:[infos[6]], cols[7]:[infos[7]]}
info_gain_curve = pd.DataFrame(info_gain_curve)

#print(info_gain_int)
#print(info_gain_curve)

# Various classification models
log_clf = LogisticRegression(solver="liblinear", random_state=42)
svm_clf = SVC(gamma="auto", probability=True, random_state=42)
tree_clf = DecisionTreeClassifier(random_state=42)
bag_clf = skle.BaggingClassifier(tree_clf, n_estimators=500, max_samples=100,
                                 bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
rnd_clf = skle.RandomForestClassifier(n_estimators=500, random_state=42)

voting_hard_clf = skle.VotingClassifier(estimators=[("lr", log_clf),
                                        ("rf", rnd_clf), ("svc", svm_clf),
                                        ("bag", bag_clf), ("tree", tree_clf)],
                                        voting="hard")
voting_hard_clf.fit(X_train, y_train)

voting_soft_clf = skle.VotingClassifier(estimators=[("lr", log_clf),
                                        ("rf", rnd_clf), ("svc", svm_clf),
                                        ("bag", bag_clf), ("tree", tree_clf)],
                                        voting="soft")
voting_soft_clf.fit(X_train, y_train)

ada_clf = skle.AdaBoostClassifier(tree_clf, n_estimators=200, algorithm="SAMME.R",
                                  learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)

gd_clf = skle.GradientBoostingClassifier(loss="exponential", n_estimators=500,
                                         random_state=42)
gd_clf.fit(X_train, y_train)

xg_clf = xgb.XGBClassifier(objective="multi:softprob", num_class=2, max_depth=5,
                           n_estimators=200, learning_rate=0.1, random_state=42)
xg_clf.fit(X_train, y_train)

voting_all_clf = skle.VotingClassifier(estimators=[("lr", log_clf),
                            ("rf", rnd_clf), ("svc", svm_clf),
                            ("bag", bag_clf), ("tree", tree_clf),
                            ("ada", ada_clf), ("gd", gd_clf), ("xg", xg_clf)],
                            voting="soft", n_jobs=-1)
voting_all_clf.fit(X_train, y_train)

def classifier(clf, title):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mse = sklm.mean_squared_error(y_test, y_pred)
    accuracy = sklm.accuracy_score(y_test, y_pred)
    variance = np.mean(np.var(y_pred))
    bias = np.mean((y_test - np.mean(y_pred))**2)

    cr = sklm.classification_report(y_test, y_pred)
    cks = sklm.cohen_kappa_score(y_test, y_pred)

    print("Classification report for " + title + " : \n", cr)

    print("Cross-validation score:")  # Test score
    print(cross_val_score(clf, X_test, y_test, cv=10))

    score_and_mse = {"Model":[title], "Score":[accuracy],
                     "Cohen Kappa Score":[cks], "MSE":[mse], "Var":[variance],
                     "Bias":[bias]}
    score_and_mse = pd.DataFrame(score_and_mse)

    conf_mat = skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    conf_mat.set_title("Norm. Confusion Matrix:\n" + title)

    y_probas = clf.predict_proba(X_test)
    rocplot = skplt.metrics.plot_roc(y_test, y_probas)
    rocplot.set_title("ROC curves:\n" + title)

    cumu_gain = skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    cumu_gain.set_title("Cumulative Gains Curve:\n" + title)

    print(score_and_mse)

def data_info():
    pulsar_data.info()
    print(pulsar_data.head())
    print("Value|Count")
    print(pulsar_data["Target"].value_counts())
    f, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(pulsar_data.corr(), annot=True, linecolor="blue", fmt=".2f", ax=ax)

def create_tree():
    tree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    tree_clf.fit(X, y)

    export_graphviz(tree_clf, out_file="Data/tree2.dot", filled=True, rounded=True,
                    special_characters=True, proportion=False,
                    class_names="Target", feature_names=list(feature.columns))
    cmd = 'dot -Tpng Data/tree2.dot -o Data/tree2.png'
    os.system(cmd)

def xbg_plot():
    xgb.plot_tree(xg_clf, num_trees=0)
    plt.rcParams["figure.figsize"] = [50, 15]
    plt.show()

    xgb.plot_importance(xg_clf)
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.show()

if __name__ == "__main__":
    #data_info()
    #create_tree()
    #classifier(log_clf, title="Logistic Regression")
    #classifier(svm_clf, title="Support Vector Machine")
    #classifier(tree_clf, title="Decision Tree")
    #classifier(bag_clf, title="Bagging")
    #classifier(rnd_clf, title="Random Forest")
    #classifier(voting_hard_clf, title="Hard Voting Class")
    #classifier(voting_soft_clf, title="Soft Voting Class")
    #classifier(ada_clf, title="AdaBoost")
    #classifier(gd_clf, title="Gradient boost")
    #classifier(xg_clf, title="XGBoost")
    classifier(voting_all_clf, title="Voting All")
    #xbg_plot()
    pass
