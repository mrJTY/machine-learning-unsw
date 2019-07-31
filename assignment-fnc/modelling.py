import config

def train_model(model_name, train_X, train_Y):
    model = config.MODELS[model_name]
    model.fit(train_X, train_Y)
    return model

def test_model(model, test_X, test_Y):
    score = model.score(test_X, test_Y)
    return score

def create_and_score_model(model_name, train_X, train_Y, test_X, test_Y):
    print(f"Training a {model_name} model")
    model = train_model(model_name, train_X, train_Y)
    score = test_model(model, test_X, test_Y)

    print(f"Trained a model using {model}")
    print(f"Got an accuracy score of {score}")


