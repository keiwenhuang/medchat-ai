import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    '''clean and preprocess data'''
    return df


def split_data(df, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state)


def save_processed_data(train_df, val_df, train_path, val_path):
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)


if __name__ == '__main__':
    raw_data_path = 'data/raw/train.csv'
    train_data_path = 'data/processed/train_data.csv'
    val_data_path = 'data/processed/val_data.csv'

    df = load_data(raw_data_path)
    df_processed = preprocess_data(df)
    train_df, val_df = split_data(df_processed)
    save_processed_data(train_df, val_df, train_data_path, val_data_path)

    print('Data preparation completed!')
