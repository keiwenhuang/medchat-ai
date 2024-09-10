from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from typing import List, Dict
import os
from dotenv import load_dotenv


class BaseModel:
    def __init__(self, model_name: str = 'gpt-3.5-turbo'):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

        self.model = ChatOpenAI(model=model_name)

    def generate_response(self, user_input: str, conversation_history: List[Dict] = None) -> str:
        messages = []
        if conversation_history:
            for message in conversation_history:
                if message['role'] == 'human':
                    messages.append(HumanMessage(content=message['content']))
                elif message['role'] == 'ai':
                    messages.append(AIMessage(content=message['content']))

        messages.append(HumanMessage(content=user_input))
        response = self.model.invoke(messages)

        return response.content

    def process_input(self, user_input: str) -> str:
        # add any preprocessing steps for the user input
        return user_input

    def run_conversation(self, user_input: str, conversation_history: List[Dict] = None) -> str:
        processed_input = self.process_input(user_input)
        response = self.generate_response(
            processed_input, conversation_history)
        return response
