import numpy as np
import pandas as pd
from main import LogisticRegression, MultilayerPerceptronClassifier, Log10Uniform, MultilayerPerceptronRegressor
from sklearn.model_selection import RandomizedSearchCV
import sys

np.random.seed(777)

train_set = np.load("Data/Credit_train.npz")
#test_set = np.load("Data/Credit_test.npz")

X_train, y_train = train_set["X_train"], train_set["y_train"].reshape(-1, 1)
#X_test, y_test = test_set["X_test"], test_set["y_test"].reshape(-1, 1)

def train_LR():
    LR = LogisticRegression(n_epochs=1000, rtol=0.01, batch_size="auto")

    possible_learning_rates = Log10Uniform(-5, -2)
    param_dist_LR = {"learning_rate": possible_learning_rates}

    # Use Randomized search for hyperparameters
    random_search_LR = RandomizedSearchCV(LR, param_distributions=param_dist_LR,
                                      n_iter=100, n_jobs=-1, iid=False, cv=5,
                                      verbose=True, error_score=0,
                                      return_train_score=True)

    # Fit the trained data
    random_search_LR.fit(X_train, y_train)

    # Export the trained data
    random_search_LR.best_estimator_.save_model("LR_credit_model.npz")

    df_LR = pd.DataFrame.from_dict(random_search_LR.cv_results_, orient="index")
    df_LR.to_csv("Data/train_credit_LR.csv")


def train_MLP():
    MLP = MultilayerPerceptronClassifier(n_epochs=300, batch_size="auto",
                                         hidden_layer_size=[100, 50], rtol=1e-2)

    learning_rates = Log10Uniform(-3, -1)
    lambdas = Log10Uniform(-10, 0)
    param_dist_MLP = {"learning_rate": learning_rates, "lambd": lambdas}

    # Use Randomized search for hyperparameters
    random_search_MLP = RandomizedSearchCV(MLP, param_distributions=param_dist_MLP,
                                      n_iter=100, n_jobs=-1, iid=False, cv=5,
                                      verbose=True, error_score=0,
                                      return_train_score=True)

    # Fit the trained data
    random_search_MLP.fit(X_train, y_train)

    # Export the trained data
    random_search_MLP.best_estimator_.save_model("NN_credit_model.npz")

    df_MLP = pd.DataFrame.from_dict(random_search_MLP.cv_results_, orient="index")
    df_MLP.to_csv("Data/train_credit_NN.csv")



def train_reg():
    nx = 50
    ny = 50
    sigma = 0.2

    def r2_scorer_fix_nan(regressor, X, y):
        y_pred = regressor.predict(X)
        if np.any(np.isnan(y_pred)):
            return -1
        else:
            return regressor.r2_score(X, y)

    train_set = np.load(f"Data/Franke_train_{nx}_{ny}_{sigma}.npz")
    #test_set = np.load(f"Data/Franke_test_{nx}_{ny}_{sigma}.npz")

    X_train, z_train = train_set["X_train"], train_set["z_train"].reshape(-1, 1)
    #X_test, z_test = test_set["X_test"], test_set["z_test"].reshape(-1, 1)


    reg = MultilayerPerceptronRegressor(
            n_epochs=300,
            batch_size="auto",
            hidden_layer_size=[100, 50],
            rtol=1e-2,
            verbose=False,
            activation_function_output="linear",)

    learning_rates = Log10Uniform(-5, -2)
    lambdas = Log10Uniform(-10, -1)
    param_dist_reg = {"learning_rate": learning_rates, "lambd": lambdas}

    random_search_reg = RandomizedSearchCV(
            reg,
            n_iter=100,
            scoring=r2_scorer_fix_nan,
            param_distributions=param_dist_reg,
            cv=5,
            iid=False,
            n_jobs=-1,
            verbose=True,
            return_train_score=True,
            error_score=np.nan)

    random_search_reg.fit(X_train, z_train)
    random_search_reg.best_estimator_.save_model(f"Reg_Franke_model_{nx}_{ny}_{sigma}.npz")
    df_results = pd.DataFrame.from_dict(random_search_reg.cv_results_, orient="index")
    df_results.to_csv(f"Data/train_franke_NN_{nx}_{ny}_{sigma}.csv")

if __name__ == "__main__":
    try:
        train = str(sys.argv[1])
    except IndexError:
        raise IndexError(f"Input the desired function for training data")
    except ValueError:
        raise TypeError("Input must be string; LR, MLP or Reg")

    if train == "LR":
        print("Training LR")
        train_LR()
    elif train == "MLP":
        print("Training MLP")
        train_MLP()
    elif train == "Reg":
        print("Training Reg")
        train_reg()
    else:
        pass
