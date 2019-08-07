from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
import config
from sklearn.metrics import accuracy_score
import fnc_challenge_utils.scoring as scoring
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
import time
from sklearn.ensemble import RandomForestClassifier

def rf():
    return RandomForestClassifier(random_state=123, n_estimators=200)

def adaboost():
    return AdaBoostClassifier(random_state=123, n_estimators=200)

def nb():
    return MultinomialNB()

def nnet():
    clf= MLPClassifier(solver='adam', hidden_layer_sizes=(120, 120, 120), random_state=123, activation='relu', learning_rate='adaptive', learning_rate_init=0.001, alpha=0.01, verbose=True)
    clf.out_activation_ = 'softmax'
    return clf

def simple_decision_tree():
    """
    Simple tree with default parameters
    Expect to have the lowest score
    """
    return DecisionTreeClassifier()

def random_cv_tree():
    """
    Use a randomised cross validation search to find the
    best parameters for a Decision Tree
    """
    n_iter_search = 10
    clf = DecisionTreeClassifier(random_state=123)
    param_distributions = {
        'min_samples_leaf': [2, 10, 50, 100],
        'max_depth': [3, 5, 10],
        'min_samples_split': [3, 5, 10],
        'max_features': ["sqrt"]
    }
    return RandomizedSearchCV(clf, n_iter=n_iter_search, cv=3, iid=False, param_distributions=param_distributions)

def gbm():
    """
    Gradient boosting model
    This was the baseline of the FNC challenge
    """
    return GradientBoostingClassifier(n_estimators=200)


MODELS = {
    'tree': simple_decision_tree,
    'random_tree': random_cv_tree,
    'gbm': gbm,
    'nnet': nnet,
    'nb': nb,
    'adaboost': adaboost,
    'rf': rf
}

def train_sklearn_model(model_name, train_X, train_Y, test_X, test_Y):
    train_X = StandardScaler().fit_transform(train_X)
    test_X = StandardScaler().fit_transform(test_X)

    print("")
    print(f"Training a {model_name} model")
    print("")
    model = MODELS[model_name]()
    start_time = time.time()
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
    return model
