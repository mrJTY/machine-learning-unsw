import pandas as pd
import feature_engineering as fe

def read_comp_data():
    train_bodies = pd.read_csv("fnc-1/train_bodies.csv")
    train_stances = pd.read_csv("fnc-1/train_stances.csv")
    test_comp_bodies = pd.read_csv("fnc-1/competition_test_bodies.csv")
    test_comp_stances = pd.read_csv("fnc-1/competition_test_stances.csv")

    return (train_bodies, train_stances, test_comp_bodies, test_comp_stances)

def merge_stance_and_body(stance_df, body_df):
    return pd.merge(left=stance_df, right=body_df, left_on="Body ID", right_on="Body ID")

def merge_refutes(df):
    """
    Assumes that df already has headline and article body
    """
    count_refutes = pd.Series(fe.refuting_features(df['Headline'], df['articleBody']))
    count_refutes.name = 'count_refutes'

    return pd.concat([df, count_refutes], axis=1)

def preprocess_data():
    train_bodies, train_stances, test_comp_bodies, test_comp_stances = read_comp_data()
    train = merge_stance_and_body(train_stances, train_bodies)
    test_comp = merge_stance_and_body(test_comp_stances, test_comp_bodies)

    train = merge_refutes(train)
    test = merge_refutes(train)

    return train, test



if __name__ == '__main__':
    pass
