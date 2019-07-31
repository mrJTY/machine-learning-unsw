import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction
import scipy.sparse as sp
import pickle
import re
import nltk
import numpy as np
import config
import pdb

# Sourced from: https://github.com/FakeNewsChallenge/fnc-1-baseline
_wnl = nltk.WordNetLemmatizer()

def normalize_word(w):
    """
    Lower case a word
    """
    return _wnl.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def refuting_features(headlines, bodies):
    """
    Sourced from FNC utilities function
    Returns a pd.Series of the count of refuting words
    """
    refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny',
        'denies',
        'refute',
        'not',
        'despite',
        'nope',
        'doubt',
        'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in enumerate(zip(headlines, bodies)):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in refuting_words]
        # Return a count of refuting words
        features = sum(features)
        X.append(features)
    X = np.array([X]).T
    return X

def remove_stopwords(l):
    """
    Sourced from FNC utilities function
    Removes stopwords from a list of tokens
    """
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def read_comp_data(datasources, train_key = 'train', test_key = 'test'):
    """
    Read the raw competition data
    """
    train_bodies = pd.read_csv(datasources[train_key]['bodies'])
    train_stances = pd.read_csv(datasources[train_key]['stances'])
    test_bodies = pd.read_csv(datasources[test_key]['bodies'])
    test_stances = pd.read_csv(datasources[test_key]['stances'])

    return (train_bodies, train_stances, test_bodies, test_stances)

def clean_bodies(df):
    """
    Return a cleaned up version of the given dataframe
    """
    def cleaner(body):
        return get_tokenized_lemmas(clean(body))

    df['articleBody'] = df['articleBody'].apply(lambda row: cleaner(body))

    return df


def merge_stance_and_body(stance_df, body_df):
    return pd.merge(left=stance_df, right=body_df, left_on="Body ID", right_on="Body ID")

def count_refutes(df):
    """
    Assumes that df already has headline and article body
    """
    count_refutes = refuting_features(df['Headline'], df['articleBody'])
    return count_refutes

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

def preprocess_data(datasources, train_key = 'train', test_key = 'test'):
    """
    Read from input data and save as matrix pickles
    """

    print("Reading data..")
    train_bodies, train_stances, test_bodies, test_stances = read_comp_data(datasources, train_key, test_key)
    train = merge_stance_and_body(train_stances, train_bodies)
    test = merge_stance_and_body(test_stances, test_bodies)
    print(f"Train shape : {train.shape}")
    print(f"Test shape : {test.shape}")
    print("")

    # Create a vectorizer from both the train and test sets
    print("Fitting a tfidf vectorizer..")
    tfidf = tfvectorizer(train, test)
    train_words = tfidf.transform(train['articleBody'])
    test_words = tfidf.transform(test['articleBody'])
    print(f"Tfidf vectorized train_words shape: {train_words.shape}")
    print(f"Tfidf vectorized test_words shape: {test_words.shape}")
    print("")

    # Count refutes
    print("Counting the number of refutes...")
    train_refutes = count_refutes(train)
    test_refutes = count_refutes(test)
    print(f"Train refutes shape {train_refutes.shape}")
    print(f"Test refutes shape {test_refutes.shape}")
    print("")

    # Create the X features and Y labels
    train_X = sp.hstack((train_refutes, train_words))
    train_Y = train['Stance'].apply(lambda key: config.LABELS[key])

    # Create the X features and Y labels
    test_X = sp.hstack((test_refutes, test_words))
    test_Y = test['Stance'].apply(lambda key: config.LABELS[key])

    write_to_pickle(train_X, train_Y, test_X, test_Y)



def write_to_pickle(train_X, train_Y, test_X, test_Y):
    """
    Dump to a pickle for faster load
    """
    pickle.dump(train_X, open("data/train_X.pickle", "wb"))
    pickle.dump(train_Y, open("data/train_Y.pickle", "wb"))
    pickle.dump(test_X, open("data/test_X.pickle", "wb"))
    pickle.dump(test_Y, open("data/test_Y.pickle", "wb"))

def load_pickles():
    """
    Dump to a pickle for faster load
    """
    return (pickle.load(open("data/train_X.pickle", "rb")),
            pickle.load(open("data/train_Y.pickle", "rb")),
            pickle.load(open("data/test_X.pickle", "rb")),
            pickle.load(open("data/test_Y.pickle", "rb")))
