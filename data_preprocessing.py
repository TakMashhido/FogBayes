import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path, encoding='ISO-8859-1'):
    df = pd.read_csv(path, encoding=encoding)
    return df

def clean_data(df):
    df = df.fillna(df.median(numeric_only=True))
    return df

def split_data(df, features, target, test_size, seed):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
