import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate

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

# Before scaling
print("Train set accuracy with Decision Tree: {:.6f}".format(tree_clf.score(X,y)))



from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,mean_squared_error,confusion_matrix,classification_report
from sklearn.metrics import cohen_kappa_score, roc_curve

# Split into training and test data
y = np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)

# Scale training and test design matrix data
scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

# Various classification models
log_clf = LogisticRegression(solver="liblinear", random_state=42)
svm_clf = SVC(gamma="auto", probability=True, random_state=42)
tree_clf = DecisionTreeClassifier(random_state=42)
bag_clf = BaggingClassifier(tree_clf, n_estimators=500, max_samples=100,
                            bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)

voting_hard_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf),
                                               ('svc', svm_clf), ('bag', bag_clf),
                                               ('tree', tree_clf)],
                                    voting='hard')
voting_hard_clf.fit(X_train, y_train)

voting_soft_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf),
                                               ('svc', svm_clf), ('bag', bag_clf),
                                               ('tree', tree_clf)],
                                    voting='soft')
voting_soft_clf.fit(X_train, y_train)
"""
for clf in (log_clf, rnd_clf, svm_clf, voting_clf, tree_clf, bag_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", probability=True, random_state=42)


for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
"""
def classifier(clf, title):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=["P", "N"])
    cm["total"] = cm[0]+cm[1]
    cm.columns = ["P", "N", "total"]
    cr = classification_report(y_test, y_pred)
    cks = cohen_kappa_score(y_test, y_pred)

    print("Cross-validation score:")
    print(cross_val_score(tree_clf, X_test, y_test, cv=10))

    score_and_mse = {"Model":[title], "Score":[accuracy],
                     "Cohen Kappa Score":[cks], "MSE":[mse]}
    score_and_mse = pd.DataFrame(score_and_mse)

    print("Classification report for " + title + " : \n", cr)

    f, axes = plt.subplots(figsize=(10,6))
    g1 = sns.heatmap(cm, annot=True, fmt=".1f", cmap="flag", ax=axes)
    g1.set_xlabel("y_head")
    g1.set_ylabel("y_true")
    g1.set_title("Confusion Matrix: " + title)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot([0,1], [0,1], "k--", color="grey")
    plt.plot(fpr, tpr, "r")
    plt.title("ROC curve: " + title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    print(score_and_mse)

import scikitplot as skplt
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
ada_clf = AdaBoostClassifier(tree_clf, n_estimators=200, algorithm="SAMME.R",
                                 learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)

gd_clf  = GradientBoostingClassifier(random_state=42)
gd_clf.fit(X_train, y_train)

xg_clf = xgb.XGBClassifier(random_state=42)
xg_clf.fit(X_train, y_train)
def boosting(clf):
    print("Cross-validation score:")
    print(cross_validate(clf, X_test, y_test, cv=10)["test_score"])
    y_pred = clf.predict(X_test)
    scores = clf.score(X_test, y_test)
    print("Test accuracy score = ", scores)
    mse = np.mean(np.mean((y_test-y_pred)**2))
    print("Mean squared error", mse)

    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    plt.show()
    y_probas = clf.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, y_probas)
    plt.show()
    skplt.metrics.plot_cumulative_gain(y_test, y_probas)
    plt.show()

def xbg_plot():
    xgb.plot_tree(xg_clf, num_trees=0)
    plt.rcParams["figure.figsize"] = [50, 15]
    plt.show()

    xgb.plot_importance(xg_clf)
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.show()

#classifier(log_clf, title="Logistic Regression")
#classifier(svm_clf, title="Support Vector Machine")
#classifier(tree_clf, title="Decision Tree")
#classifier(bag_clf, title="Bagging")
#classifier(rnd_clf, title="Random Forest")
#classifier(voting_hard_clf, title="Hard Voting Classifier")
#classifier(voting_soft_clf, title="Soft Voting Classifier")
#boosting(ada_clf)
#boosting(gd_clf)
#boosting(xg_clf)
#xbg_plot()


