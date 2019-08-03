from keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

def nnet_keras(train_X, train_Y):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))
    model.fit(train_X, train_Y, epochs=5, batch_size=32)
    return model

def decision_tree(train_X, train_Y):
    model = DecisionTreeClassifier()
    model.fit(train_X, train_Y)
    return model

MODELS = {
    'tree': decision_tree,
    'nnet': nnet_keras
}

def train_model(model_name, train_X, train_Y, test_X, test_Y):
    print(f"Training a {model_name} model")
    model = MODELS[model_name](train_X, train_Y)

    print(f"Trained a model using {model}")
    return model
