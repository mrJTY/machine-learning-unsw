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
import pdb


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
        print(f"Sampling the dataset with only {n_samples} samples")
        merged = merged.sample(n=n_samples, random_state=123)

    return merged

def count_refutes(train_df, test_df):
    """
    Count the number of times a refusal word was seen in the headline
    Assumes that df already has headline and article body
    """
    print("Counting the number of refutes...")
    train_X = fe.refuting_features(train_df['Headline'], train_df['articleBody'])
    test_X = fe.refuting_features(test_df['Headline'], test_df['articleBody'])
    print(f"Train refutes shape {train_X.shape}")
    print(f"Test refutes shape {test_X.shape}")
    print("")
    return train_X, test_X

def count_overlaps(train_df, test_df):
    """
    Count the number of times
    Assumes that df already has headline and article body
    """
    print("Counting the number of overlaps...")
    train_X = fe.word_overlap_features(train_df['Headline'], train_df['articleBody'])
    test_X = fe.word_overlap_features(test_df['Headline'], test_df['articleBody'])
    print(f"Train refutes shape {train_X.shape}")
    print(f"Test refutes shape {test_X.shape}")
    print("")
    return train_X, test_X

def count_polarity(train_df, test_df):
    """
    Count the number of times
    Assumes that df already has headline and article body
    """
    print("Counting the number of polarity...")
    train_X = fe.word_overlap_features(train_df['Headline'], train_df['articleBody'])
    test_X = fe.word_overlap_features(test_df['Headline'], test_df['articleBody'])
    print(f"Train refutes shape {train_X.shape}")
    print(f"Test refutes shape {test_X.shape}")
    print("")
    return train_X, test_X

def count_hand(train_df, test_df):
    """
    Count the number of times
    Assumes that df already has headline and article body
    """
    print("Counting the number of hand...")
    train_X = fe.word_overlap_features(train_df['Headline'], train_df['articleBody'])
    test_X = fe.word_overlap_features(test_df['Headline'], test_df['articleBody'])
    print(f"Train refutes shape {train_X.shape}")
    print(f"Test refutes shape {test_X.shape}")
    print("")
    return train_X, test_X


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
    svd = TruncatedSVD(n_components=500, n_iter=7, random_state=123)
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

    # TfIDF Vectorize
    train_words, test_words = create_tfidf_matrix(train, test)
    train_bag_of_words, explained_var_train = reduce_dimensions(train_words)
    print("Explained variance of SVD on train features: {}%".format(int(explained_var_train * 100)))
    test_bag_of_words, explained_var_test = reduce_dimensions(test_words)
    print("Explained variance of SVD on test features: {}%".format(int(explained_var_test * 100)))

    # Count refutes
    train_refutes, test_refutes = count_refutes(train, test)

    # Count overlap
    train_overlaps, test_overlaps = count_overlaps(train, test)

    # Count polarity
    train_polarity, test_polarity = count_polarity(train, test)

    # Count hand
    train_hand, test_hand = count_hand(train, test)

    # Create the X features and Y labels
    train_X = np.hstack([train_bag_of_words, train_refutes, train_overlaps, train_polarity, train_hand])
    train_Y = train['Stance'].apply(lambda key: config.LABEL_LOOKUP[key])

    # Create the X features and Y labels
    test_X = np.hstack([test_bag_of_words, test_refutes, test_overlaps, test_polarity, test_hand])
    test_Y = test['Stance'].apply(lambda key: config.LABEL_LOOKUP[key])

    write_to_pickle(train_X, train_Y, test_X, test_Y, train_prop)



