from models.base_model import BaseModel
import pandas as pd
import argparse


def load_sample_data(file_path: str, num_samples: int = 5):
    df = pd.read_csv(file_path)
    return df.sample(n=num_samples)


def main(data_file):
    model = BaseModel()
    samples = load_sample_data(data_file, 1)

    for _, row in samples.iterrows():
        question = row['Question']
        print(f"Question: {question}\n")
        response = model.run_conversation(question)
        print(f"Model response: {response}\n")
        print(f"Actual answer: {row['Answer']}\n")
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the base model on healthcare data.")
    parser.add_argument('--data_file', type=str, default='data/raw/train.csv',
                        help='Path to the CSV file containing healthcare Q&A data')
    args = parser.parse_args()

    main(args.data_file)
