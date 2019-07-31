import preprocess as pe
import modelling as mo
import config
import argparse

def parse_args():
    parser = argparse.ArgumentParser();
    parser.add_argument("--process_data", help="If flag is true, then it will process and vectorize words from the train and test sets. Otherwise, it will load .pickle files", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args();

    if args.process_data:
         pe.preprocess_data(config.DATASOURCES)

    train_X, train_Y, test_X, test_Y = pe.load_pickles()

    mo.create_and_score_model('gradient_boost', train_X, train_Y, test_X, test_Y)

#    model.fit(train_X, train_Y)
#    train_score = model.score(train_X, train_Y)
#    print(train_score)
#
#
#
#    pickle.dump(train_refutes_words, open( "pickles/train.pickle", "wb" ) )
#
#    print("Done")
