import config

# def test_model(model, test_X, test_Y):
#     score = model.score(test_X, test_Y)
#     return score

def train_model(model_name, train_X, train_Y, test_X, test_Y):
    print(f"Training a {model_name} model")
    model = config.MODELS[model_name]
    model.fit(train_X, train_Y)

    print(f"Trained a model using {model}")
    return model


