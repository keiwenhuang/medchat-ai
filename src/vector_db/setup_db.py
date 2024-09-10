import json
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document

PROJECT_ROOT = Path(__file__).parents[2].resolve()

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def create_documents(data):
    return [Document(page_content=item['text'], metadata={'qtype': item['qtype']}) for item in data]

def setup_vector_db(data, embedding_model_name, persist_directory):
    # initialize
    embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_name)
    
    # create documents
    documents = create_documents(data)
    
    # create and persist chroma database
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    vectordb.persist()
    print(f'Vector database created and persisted to {persist_directory}')
    return vectordb

def main():
    # load processed data
    train_data = load_json_data(f'{PROJECT_ROOT}/data/processed/train_data.json')
    
    # setup vector database
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    persist_dir = f'{PROJECT_ROOT}/data/vector_db'
    
    vectordb = setup_vector_db(train_data, embedding_model_name, persist_dir)
    
    # test a query
    query = 'What are the symptoms of Lymphocytic Choriomeningitis?'
    results = vectordb.similarity_search(query, k=1)
    print(f'\nTest query: {query}')
    print(f'Most similar document: {results[0].page_content[:200]}...')
    
if __name__ == '__main__':
    main()
    

