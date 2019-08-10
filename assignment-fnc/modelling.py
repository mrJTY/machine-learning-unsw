from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
import config
import fnc_challenge_utils.scoring as scoring
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, accuracy_score
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from learning_curve import plot_learning_curv

def simple_decision_tree():
    """
    Simple tree with default parameters
    Expect to have the lowest score
    Use this to do validation curves
    """
    return DecisionTreeClassifier()

def rf():
    return RandomForestClassifier(random_state=config.RANDOM_STATE, n_estimators=200, max_depth=config.TREE_MAX_DEPTH, min_samples_leaf=config.TREE_MIN_SAMPLES_LEAF)

def xgboost_plain():
    return XGBClassifier(n_estimators=200)

def xgboost_tuned():
    return XGBClassifier(n_estimators=200, reg_alpha=0.25, reg_lambda=1.25, max_depth=config.TREE_MAX_DEPTH, max_delta_step=10, random_state=config.RANDOM_STATE)

def fit_xgboost(model, train_X, train_Y, test_X, test_Y):
    # Split out an validation set
    eval_set = [(train_X, train_Y), (test_X, test_Y)]
    model.fit(train_X, train_Y, eval_set=eval_set, early_stopping_rounds=5, eval_metric=["mlogloss"])
    # Performance metrics specifc to XGBOOST
    # Reference: https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python
    pred = model.predict(test_X)
    results = model.evals_result()
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.savefig("img/xgboost_logloss.png")
    return model

def adaboost():
    return AdaBoostClassifier(random_state=config.RANDOM_STATE, n_estimators=200)

def nb():
    return MultinomialNB()

def nnet():
    clf= MLPClassifier(solver='adam', hidden_layer_sizes=(120, 120, 120), random_state=config.RANDOM_STATE, activation='relu', learning_rate='adaptive', learning_rate_init=0.001, alpha=0.1, verbose=True)
    clf.out_activation_ = 'softmax'
    return clf

def gbm():
    """
    Gradient boosting model
    This was the baseline of the FNC challenge
    """
    return GradientBoostingClassifier(n_estimators=200)

def gbm_tune():
    """
    Gradient boosting model
    This was the baseline of the FNC challenge
    """
    return GradientBoostingClassifier(n_estimators=200, max_features='sqrt', max_depth=config.TREE_MAX_DEPTH, min_samples_leaf=config.TREE_MIN_SAMPLES_LEAF)

MODELS = {
    'tree': simple_decision_tree,
    'gbm': gbm,
    'gbm_tune': gbm_tune,
    'nnet': nnet,
    'nb': nb,
    'adaboost': adaboost,
    'rf': rf,
    'xgboost_plain': xgboost_plain,
    'xgboost_tuned': xgboost_tuned
}

def train_sklearn_model(model_name, train_X, train_Y, test_X, test_Y):
    train_X = StandardScaler().fit_transform(train_X)
    test_X = StandardScaler().fit_transform(test_X)

    print("")
    print(f"Training a {model_name} model")
    print("")
    start_time = time.time()
    model = MODELS[model_name]()

    if model_name == "xgboost_plain" or model_name == "xgboost_tuned":
        # Xgboost has a custom fit
        model = fit_xgboost(model, train_X, train_Y, test_X, test_Y)
        # Plot some validation plots on the plain xgboost
        if model_name == "xgboost_plain":
            # Tips on fine tuning xgboost: https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
            # Regularisation params
            plot_validation_curve(model, model_name, train_X, train_Y, "reg_alpha", np.linspace(0.01, 0.9, 4))
            plot_validation_curve(model, model_name, train_X, train_Y, "gamma", [0, 1, 5])
            # % of rows used
            plot_validation_curve(model, model_name, train_X, train_Y, "subsample", np.linspace(0.5, 1.0, 4))
            # % of columns used
            plot_validation_curve(model, model_name, train_X, train_Y, "colsample_bytree", np.linspace(0.3, 1.0, 4))
    elif model_name == "tree":
        # Plot some validation curves on the parameters
        model.fit(train_X, train_Y)
        plot_validation_curve(model, model_name, train_X, train_Y, "max_depth", np.linspace(3, 10, 3))
        plot_validation_curve(model, model_name, train_X, train_Y, "min_samples_split", [int(i) for i in np.linspace(2, 50, 4)])
        plot_validation_curve(model, model_name, train_X, train_Y, "min_samples_leaf", [int(i) for i in np.linspace(2, 100, 4)])
    else:
        model.fit(train_X, train_Y)

    print(f"Training time took {time.time() - start_time} seconds")
    print("")
    print(f"Trained a model using {model}")

    # Actual labels
    train_Y_labels = [config.LABELS[int(a)] for a in train_Y]
    test_Y_labels = [config.LABELS[int(a)] for a in test_Y]

    # Prediction labels
    train_pred = [config.LABELS[int(a)] for a in model.predict(train_X)]
    test_pred = [config.LABELS[int(a)] for a in model.predict(test_X)]

    # Scoring from the FNC challenge
    print("Train FNC Score:")
    train_score = scoring.report_score(train_Y_labels, train_pred)
    f1_train_score = f1_score(train_Y_labels, train_pred, average="macro")
    print(f"Train F1 Score: {f1_train_score}")
    acc_train_score = accuracy_score(train_Y_labels, train_pred)
    print(f"Train Accuracy Score: {acc_train_score}")
    print("")

    print("Test FNC Score:")
    test_score = scoring.report_score(test_Y_labels, test_pred)
    f1_test_score = f1_score(test_Y_labels, test_pred, average="macro")
    print(f"Test F1 Score: {f1_test_score}")
    acc_test_score = accuracy_score(test_Y_labels, test_pred)
    print(f"Test Accuracy Score: {acc_test_score}")

    print("Plotting learning plots...")
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    plot_learning_curve(model, model_name, train_X, train_Y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    return model
