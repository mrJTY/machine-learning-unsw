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
import time

import lightgbm as lgb

def lightgbm_model(train_X, train_Y, test_X, test_Y):
    param = {'num_leaves': 31, 'objective': 'binary'}
    param['metric'] = 'auc'
    num_round = 10
    train_data = lgb.Dataset(train_X, label=train_Y)
    best_model = lgb.train(param, train_data, num_round)

    lgb.cv(param, train_data, num_round, nfold=5)

    pred_Y = best_model.predict(test_X)
    import pdb
    pdb.set_trace()


def adaboost():
    return AdaBoostClassifier(random_state=123)

def nb():
    return MultinomialNB()

def nnet():
    return MLPClassifier(solver='adam', hidden_layer_sizes=(3,3), random_state=123, verbose=True
    )

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
    return GradientBoostingClassifier()


MODELS = {
    'tree': simple_decision_tree,
    'random_tree': random_cv_tree,
    'gbm': gbm,
    'nnet': nnet,
    'nb': nb,
    'adaboost': adaboost
}

def train_sklearn_model(model_name, train_X, train_Y, test_X, test_Y):
    start_time = time.time()
    print("")
    print(f"Training a {model_name} model")
    print("")
    model = MODELS[model_name]()
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
    print("Training score:")
    train_score = scoring.report_score(train_Y_labels, train_pred)
    print("")
    print("Test Score:")
    test_score = scoring.report_score(test_Y_labels, test_pred)

    return model
