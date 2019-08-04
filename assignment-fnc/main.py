import preprocess as pe
import modelling as mo
import config
import argparse
import fnc_challenge_utils.scoring as scoring

def parse_args():
    parser = argparse.ArgumentParser();
    parser.add_argument("--skip_preprocess", help="If flag is true, it will skip preprocessing the data and load .pickle files in the data/ folder", action="store_true")
    parser.add_argument("--model", help="Model name", required=True)
    parser.add_argument("--train_prop", help="The proportion of training dataset to read from. Between 0 and 1", default=1)
    parser.add_argument("--test_prop", help="The proportion of training dataset to read from. Between 0 and 1", default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args();
    print(args)

    train_prop = float(args.train_prop)
    test_prop = float(args.test_prop)

    if args.skip_preprocess:
        pass
    else:
         pe.preprocess_data(datasources=config.DATASOURCES, train_prop=train_prop, test_prop=test_prop)

    train_X, train_Y, test_X, test_Y = pe.load_pickles()

    clf = mo.train_model(args.model, train_X, train_Y, test_X, test_Y)

    predicted = [config.LABELS[int(a)] for a in clf.predict(test_X)]
    actual = [config.LABELS[int(a)] for a in test_Y]
    score = scoring.report_score(actual, predicted)

