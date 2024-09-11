import pandas as pd
import json
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    df["text"] = "Question: " + df["Question"] + "\nAnswer: " + df["Answer"]
    return df[["qtype", "text"]]


def split_data(df, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state)


# def save_processed_data(train_df, val_df, train_path, val_path):
#     train_df.to_csv(train_path, index=False)
#     val_df.to_csv(val_path, index=False)


def save_to_json(df, file_path):
    data = df.to_dict(orient="records")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    # Load data
    df = load_data("data/raw/healthcare_qa_dataset.csv")

    # Preprocess data
    df_processed = preprocess_data(df)

    # Split data
    train_df, val_df = split_data(df_processed)

    # Save processed data
    save_to_json(train_df, "data/processed/train_data.json")
    save_to_json(val_df, "data/processed/val_data.json")

    print(f"Processed {len(df)} rows of data.")
    print(
        f"Saved {len(train_df)} training samples and {len(val_df)} validation samples."
    )


if __name__ == "__main__":
    main()
