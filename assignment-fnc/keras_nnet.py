from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
import numpy as np
import config
import fnc_challenge_utils.scoring as scoring
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import keras.backend as K


def keras_nnet(train_X, train_Y, test_X, test_Y):
    train_X = StandardScaler().fit_transform(train_X)
    test_X = StandardScaler().fit_transform(test_X)
    num_classes = 4
    model = Sequential()
    model.add(Dense(units=512, input_dim=train_X.shape[1]))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=3, validation_split=0.1)

    train_Y_labels = [config.LABELS[int(a)] for a in train_Y]
    test_Y_labels = [config.LABELS[int(a)] for a in test_Y]
    #train_pred = [config.LABELS[int(a)] for a in model.predict(train_X)]
    #test_pred = [config.LABELS[int(a)] for a in model.predict(test_X)]
    train_pred = [config.LABELS[int(np.argmax(a))] for a in model.predict(train_X)]
    test_pred = [config.LABELS[int(np.argmax(a))] for a in model.predict(test_X)]

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
