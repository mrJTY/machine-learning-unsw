import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction
import scipy.sparse as sp
import re
import nltk
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier


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
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in enumerate(zip(headlines, bodies)):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
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

def read_comp_data():
    """
    Read the raw competition data
    """
    train_bodies = pd.read_csv("fnc-1/train_bodies.csv")
    train_stances = pd.read_csv("fnc-1/train_stances.csv")
    test_bodies = pd.read_csv("fnc-1/competition_test_bodies.csv")
    test_stances = pd.read_csv("fnc-1/competition_test_stances.csv")

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

labels = {
    'agree': 0,
    'disagree': 1,
    'discuss': 2,
    'unrelated': 3
}

if __name__ == '__main__':
    print("Reading data..")
    train_bodies, train_stances, test_bodies, test_stances = read_comp_data()
    train = merge_stance_and_body(train_stances, train_bodies)
    test = merge_stance_and_body(test_stances, test_bodies)

    # Count refutes
    print("Counting the number of refutes...")
    train_refutes = count_refutes(train)
    test_refutes = count_refutes(test)

    # Transform with tfidf
    print("Fitting a tfidf vectorizer..")
    tfidf = tfvectorizer(train, test)

    train_words = tfidf.transform(train['articleBody'])

    # Combine refutes and words
    train_X = sp.hstack((train_refutes, train_words))


    pdb.set_trace()
    # Train a model on this...
    train_Y = train['Stance'].apply(lambda key: labels[key])

    model = DecisionTreeClassifier()
    model.fit(train_X, train_Y)
    train_score = model.score(train_X, train_Y)



    pickle.dump(train_refutes_words, open( "pickles/train.pickle", "wb" ) )

    print("Done")
