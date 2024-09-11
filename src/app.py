from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pathlib import Path
from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).parents[2].resolve()

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable."
    )


def load_vector_db(persist_directory):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def setup_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(temperature=0)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


def main():
    persist_dir = f"{PROJECT_ROOT}/data/vector_db"
    vectordb = load_vector_db(persist_dir)
    qa_chain = setup_qa_chain(vectordb)

    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == "quit":
            break

        result = qa_chain.invoke(query)
        print(f"\nAnswer: {result}\n")


if __name__ == "__main__":
    main()
