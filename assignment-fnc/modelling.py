from keras.layers import Dense
from keras.models import Sequential
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

def nnet_keras(train_X, train_Y, num_dimensions=29265):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=num_dimensions))
    model.add(Dense(units=1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    model.fit(train_X, train_Y)
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
