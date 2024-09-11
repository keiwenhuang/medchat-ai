import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).parents[2].resolve()
print(PROJECT_ROOT)


class BaseChatBot:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable."
            )

        self.model = ChatOpenAI(model="gpt-3.5-turbo")
        self.embeddings = OpenAIEmbeddings()
        self.system_message = SystemMessage(
            content="You are a helpful AI assistant specializing in healthcare information. Provide accurate and concise answers to medical questions."
        )

        # load existing vector database
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        persist_dir = f"{PROJECT_ROOT}/data/vector_db"
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store = Chroma(
            persist_directory=persist_dir, embedding_function=self.embeddings
        )

        # initialize retrieval chain
        self.initialize_retrieval_chain()

    def initialize_retrieval_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """Answer the following question based on the given context:
        
        Context: {context}
        
        Question: {input}
        
        Answer: 
        """
        )

        document_chain = create_stuff_documents_chain(self.model, prompt)
        self.retrieval_chain = create_retrieval_chain(
            self.vector_store.as_retriever(), document_chain
        )

    def generate_response(self, user_input):
        try:
            if self.retrieval_chain:
                # perform similarity search first
                relevant_docs = self.vector_store.similarity_search_with_score(
                    user_input, k=1
                )

                if relevant_docs and relevant_docs[0][1] < 0.8:
                    response = self.retrieval_chain.invoke({"input": user_input})
                    return {"answer": response["answer"], "source": "vector_database"}
                else:
                    messages = [self.system_message, HumanMessage(content=user_input)]
                    response = self.model(messages)
                    return {"answer": response.content, "source": "base_model"}
            else:
                messages = [self.system_message, HumanMessage(content=user_input)]
                response = self.model(messages)
                return {"answer": response.content, "source": "base_model"}

        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "answer": "Sorry, Encountered an error while processing request. Please try again.",
                "source": "error",
            }

    def process_user_input(self, user_input):
        processed_input = user_input.strip().lower()
        return processed_input


if __name__ == "__main__":
    chatbot = BaseChatBot()
    print(
        "Healthcare AI Assistant: Hello! I'm here to help with your medical questions. Type 'exit' to end the conversation."
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("AI Assistant: Goodbye!")
            break

        processed_input = chatbot.process_user_input(user_input)
        response = chatbot.generate_response(processed_input)
        print(f'AI Assistant: {response["answer"]}')
        print(f'Source: {response["source"]}')
