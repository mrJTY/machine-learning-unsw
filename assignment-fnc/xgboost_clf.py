import xgboost as xgb
import time
from xgboost import XGBClassifier

def train_xgboost(train_X, train_Y, test_X, test_Y):
    print("XGBOOST!")
    #xgtrain = xgb.DMatrix(train_X.tolist(), train_Y)
    #xgtest = xgb.DMatrix()
    #num_round = 2

    print("")
    print("")
    model = XGBClassifier(objective='multi:softmax')
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
