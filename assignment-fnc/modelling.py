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


def nb():
    return MultinomialNB()

def nnet():
    return MLPClassifier(solver='adam', hidden_layer_sizes=(3, 3), random_state=123,
                          alpha=0.005, verbose=False
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
    'simple_tree': simple_decision_tree,
    'random_tree': random_cv_tree,
    'gbm': gbm,
    'nnet': nnet,
    'nb': nb
}

def train_sklearn_model(model_name, train_X, train_Y, test_X, test_Y):
    print(f"Training a {model_name} model")
    model = MODELS[model_name]()
    model.fit(train_X, train_Y)
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
    print("")


    # Diagonal scores, won't be using them
    #print(f"Train score: {accuracy_score(train_Y_labels, train_pred)}")
    #print(f"Test score: {accuracy_score(test_Y_labels, test_pred)}")
    return model
