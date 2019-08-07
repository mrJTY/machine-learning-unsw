import preprocess as pe
import modelling as mo
import config
import argparse
import fnc_challenge_utils.scoring as scoring
import numpy as np
import keras_nnet
import xgboost_clf

def parse_args():
    parser = argparse.ArgumentParser();
    parser.add_argument("--skip_preprocess", help="If flag is true, it will skip preprocessing the data and load .pickle files in the data/ folder", action="store_true")
    parser.add_argument("--model", help="Model name", default="")
    parser.add_argument("--train_prop", help="The proportion of training dataset to read from. Between 0 and 1", default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args();
    print(args)

    train_prop = float(args.train_prop)

    if args.skip_preprocess:
        pass
    else:
         # Run the preprocessing and dump as pickles
         pe.preprocess_data(datasources=config.DATASOURCES, train_prop=train_prop)


    # Load the pickles
    train_X, train_Y, test_X, test_Y = pe.load_pickles(train_prop)
    # Train the model
    if args.model == "keras_nnet":
        keras_nnet.keras_nnet(train_X, train_Y, test_X, test_Y)
    elif args.model == "xgboost":
        xgboost_clf.train_xgboost(train_X, train_Y, test_X, test_Y)

    elif args.model != "":
        mo.train_sklearn_model(args.model, train_X, train_Y, test_X, test_Y)
    else:
        print("No model trained!")


