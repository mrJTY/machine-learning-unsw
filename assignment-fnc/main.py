import preprocess as pe
import modelling as mo
import config
import argparse
import scoring

def parse_args():
    parser = argparse.ArgumentParser();
    parser.add_argument("--process_data", help="If flag is true, then it will process and vectorize words from the train and test sets. Otherwise, it will load .pickle files", action="store_true")
    parser.add_argument("--model", help="Model name", required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args();

    if args.process_data:
         pe.preprocess_data(config.DATASOURCES)

    train_X, train_Y, test_X, test_Y = pe.load_pickles()

    # TODO(JT): Add kfold
    # for fold in fold_stances:
    clf = mo.train_model('tree', train_X, train_Y, test_X, test_Y)

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


