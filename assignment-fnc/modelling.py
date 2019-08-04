from keras.layers import Dense
from keras.models import Sequential
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV


def nnet_keras(train_X, train_Y, num_dimensions=29265):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=num_dimensions))
    model.add(Dense(units=1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    model.fit(train_X, train_Y)
    return model

def simple_decision_tree(train_X, train_Y):
    """
    Simple tree with default parameters
    Expect to have the lowest score
    """
    model = DecisionTreeClassifier()
    model.fit(train_X, train_Y)
    return model

def random_cv_tree(train_X, train_Y):
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
    model = RandomizedSearchCV(clf, n_iter=n_iter_search, cv=3, iid=False, param_distributions=param_distributions)
    model.fit(train_X, train_Y)
    return model

def gbm(train_X, train_Y):
    """
    Gradient boosting model
    This was the baseline of the FNC challenge
    """

    model = GradientBoostingClassifier()
    model.fit(train_X, train_Y)
    return model


MODELS = {
    'simple_tree': simple_decision_tree,
    'random_tree': random_cv_tree,
    'gbm': gbm
}

def train_model(model_name, train_X, train_Y, test_X, test_Y):
    print(f"Training a {model_name} model")
    model = MODELS[model_name](train_X, train_Y)

    print(f"Trained a model using {model}")
    return model
