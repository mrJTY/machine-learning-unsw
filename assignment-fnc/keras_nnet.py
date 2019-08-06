from keras.layers import Dense
from keras.models import Sequential

def nnet_keras(train_X, train_Y, input_dim):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=input_dim))
    model.add(Dense(units=4, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    model.fit(train_X, train_Y)
    train_X = train_X.toarray()
    train_Y = train_Y.values
    test_X = test_X.toarray()
    test_Y = test_Y.values
    clf = mo.nnet_keras(train_X, train_Y, input_dim=train_ncols)
    predicted = [config.LABELS[int(np.argmax(a))] for a in clf.predict(test_X)]
    return model
