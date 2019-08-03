import preprocess as pe
import modelling as mo
import config
import argparse
import fnc_challenge_utils.scoring

def parse_args():
    parser = argparse.ArgumentParser();
    parser.add_argument("--process_data", help="If flag is true, then it will process and vectorize words from the train and test sets. Otherwise, it will load .pickle files", action="store_true")
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

    if args.process_data:
         pe.preprocess_data(datasources=config.DATASOURCES, train_prop=train_prop, test_prop=test_prop)

    train_X, train_Y, test_X, test_Y = pe.load_pickles()

    # TODO(JT): Add kfold
    # for fold in fold_stances:
    clf = mo.train_model(args.model, train_X, train_Y, test_X, test_Y)

    predicted = [config.LABELS[int(a)] for a in clf.predict(test_X)]
    actual = [config.LABELS[int(a)] for a in test_Y]
    score = scoring.report_score(actual, predicted)

    # TODO(JT): Find the best score
    #         fold_score, _ = scoring.score_submission(actual, predicted)
    #         max_fold_score, _ = scoring.score_submission(actual, actual)
    # 
    #         score = fold_score/max_fold_score
    # 
    #         print("Score for fold "+ str(fold) + " was - " + str(score))
    #         if score > best_score:
    #             best_score = score
    #             best_fold = clf


