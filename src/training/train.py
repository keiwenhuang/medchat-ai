import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from datasets import Dataset

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def convert_to_hf_dataset(df):
    return Dataset.from_pandas(df)


model = ChatOpenAI(model='gpt-3.5-turbo')
prompt_template = ChatPromptTemplate.from_template('You are a medical AI assistant. Answer the following {question}')
chain = prompt_template | model


def process_data(examples):
    outputs = chain.batch(examples['question'])
    return {'generated_answer': [output.content for output in outputs]}


def main():
    data = load_data('data/processed/train_data.csv')
    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = convert_to_hf_dataset(train_data)
    eval_dataset = convert_to_hf_dataset(eval_data)

    train_data = train_dataset.map(process_data, batched=True, batch_size=16)
    eval_data = eval_dataset.map(process_data, batched=True, batch_size=16)

    print('Training examples:')
    for i in range(3):
        print(f'Question: {train_dataset[i]['question']}')
        print(f'Generated Answer: {train_dataset[i]['generated_answer']}')
        print(f'Actual Answer: {train_dataset[i]['Answer']}')
        print('-' * 50)


if __name__ == '__main__':
    main()
