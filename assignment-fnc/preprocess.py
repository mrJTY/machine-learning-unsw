import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction
import scipy.sparse as sp
import pickle
import re
import nltk
import numpy as np
import config
import fnc_challenge_utils.feature_engineering as fe
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


def read_comp_data(datasources):
    """
    Read the raw competition data
    """
    train_bodies = pd.read_csv(datasources['train']['bodies'])
    train_stances = pd.read_csv(datasources['train']['stances'])
    test_bodies = pd.read_csv(datasources['test']['bodies'])
    test_stances = pd.read_csv(datasources['test']['stances'])

    return (train_bodies, train_stances, test_bodies, test_stances)

def clean_bodies(df):
    """
    Return a cleaned up version of the given dataframe
    """
    def cleaner(body):
        return get_tokenized_lemmas(clean(body))

    df['articleBody'] = df['articleBody'].apply(lambda row: cleaner(body))

    return df


def merge_stance_and_body(stance_df, body_df, prop=1):
    merged = pd.merge(left=stance_df, right=body_df, left_on="Body ID", right_on="Body ID")
    # Sample from training dataset if train_prop is given
    if prop < 1:
        n_total = merged.shape[0]
        n_samples = int(n_total * prop)
        print(f"Training the dataset with only {prop*100}% of the training set")
        merged = merged.sample(n=n_samples, random_state=123)

    return merged

def count_refutes(df):
    """
    Count the number of times a refusal word was seen in the headline
    Assumes that df already has headline and article body
    """
    X = fe.refuting_features(df['Headline'], df['articleBody'])
    return X

def count_overlaps(df):
    """
    Count the number of times
    Assumes that df already has headline and article body
    """
    X = fe.word_overlap_features(df['Headline'], df['articleBody'])
    return X

def count_polarity(df):
    """
    Count the number of times
    Assumes that df already has headline and article body
    """
    X = fe.word_overlap_features(df['Headline'], df['articleBody'])
    return X

def count_hand(df):
    """
    Count the number of times
    Assumes that df already has headline and article body
    """
    X = fe.word_overlap_features(df['Headline'], df['articleBody'])
    return X


def tfvectorizer(train, test):
    """
    Fit a tfidf vectorizer with the corpus from the train and test df's
    """
    # Get the whole set of for vectorizing the words
    corpus = pd.concat([train['articleBody'], test['articleBody']]).values
    # Term inverse freq vectorizer
    tfidf = TfidfVectorizer()
    tfidf.fit(corpus)
    return tfidf

def create_tfidf_matrix(train, test):
    # Create a vectorizer from both the train and test sets
    print("Fitting a tfidf vectorizer..")
    tfidf = tfvectorizer(train, test)
    train_words = tfidf.transform(train['articleBody'])
    test_words = tfidf.transform(test['articleBody'])
    print(f"Tfidf vectorized train_words shape: {train_words.shape}")
    print(f"Tfidf vectorized test_words shape: {test_words.shape}")
    print("")
    return train_words, test_words

def reduce_dimensions(input_matrix):
    # Recommended to be 100 for LSA
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=123)
    output = svd.fit_transform(input_matrix)
    explained_variance = svd.explained_variance_ratio_.sum()
    return output, explained_variance

def write_to_pickle(train_X, train_Y, test_X, test_Y, train_prop):
    """
    Dump to a pickle for faster load
    """
    pickle.dump(train_X, open(f"data/train_X_{train_prop}.pickle", "wb"))
    pickle.dump(train_Y, open(f"data/train_Y_{train_prop}.pickle", "wb"))
    pickle.dump(test_X, open(f"data/test_X_{train_prop}.pickle", "wb"))
    pickle.dump(test_Y, open(f"data/test_Y_{train_prop}.pickle", "wb"))

def load_pickles(train_prop):
    """
    Dump to a pickle for faster load
    """
    return (pickle.load(open(f"data/train_X_{train_prop}.pickle", "rb")),
            pickle.load(open(f"data/train_Y_{train_prop}.pickle", "rb")),
            pickle.load(open(f"data/test_X_{train_prop}.pickle", "rb")),
            pickle.load(open(f"data/test_Y_{train_prop}.pickle", "rb")))


def preprocess_features(df, tfidf_bag_of_words):
    print("Reducing dimensions of bag of words...")
    reduced_bag_of_words, explained_var = reduce_dimensions(tfidf_bag_of_words)
    print("Explained variance of SVD on features: {}%".format(int(explained_var * 100)))

    print("Calculating refutes...")
    refutes = count_refutes(df)
    print("Calculating overlaps...")
    overlaps = count_overlaps(df)
    print("Calculating polarity...")
    polarity = count_polarity(df)
    print("Calculating hand in hand cooccurence...")
    hand = count_hand(df)
    X = np.hstack([reduced_bag_of_words, refutes, overlaps, polarity, hand])

    print(f"Feature shape {X.shape}")
    print("")
    return X


def preprocess_data(datasources, train_key='train', test_key='test', train_prop=1):
    """
    Read from input data and save as matrix pickles
    """

    print("Reading data..")
    train_bodies, train_stances, test_bodies, test_stances = read_comp_data(datasources)
    print("Original data sizes")
    print(f"{len(train_bodies)}")
    print("")
    # Training set may be set to different proportions
    train = merge_stance_and_body(train_stances, train_bodies, train_prop)
    # Always read 100% of test data
    test = merge_stance_and_body(test_stances, test_bodies)
    print(f"Train shape : {train.shape}")
    print(f"Test shape : {test.shape}")
    print("")

    # TfIDF needs to be done with both train and test
    train_bag_of_words, test_bag_of_words = create_tfidf_matrix(train, test)

    # Process feature
    print("Train set processing...")
    train_X = preprocess_features(train, train_bag_of_words)
    print("Test set processing...")
    test_X = preprocess_features(test, test_bag_of_words)

    # Process labels
    train_Y = train['Stance'].apply(lambda key: config.LABEL_LOOKUP[key])
    test_Y = test['Stance'].apply(lambda key: config.LABEL_LOOKUP[key])

    write_to_pickle(train_X, train_Y, test_X, test_Y, train_prop)

