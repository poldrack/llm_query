from abc import ABC, abstractmethod
from typing import Optional, Dict
import anthropic
import openai
import os


class ChatClient(ABC):
    @abstractmethod
    def chat(self, message: str) -> str:
        pass

class OpenAIChatClient(ChatClient):
    def __init__(self, api_key: str, model: str = "gpt-4",
                 system_msg: str = None):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.system_msg = system_msg or "You are a helpful AI assistant."
        
    def chat(self, message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content

class AnthropicChatClient(ChatClient):
    def __init__(self, api_key: str, 
                 model: str = "claude-3-sonnet-20240229",
                 system_msg: str = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.system_msg = system_msg or "You are a helpful AI assistant."
    
    def chat(self, message: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.0,
            system=self.system_msg,
            messages=[{"role": "user", "content": message}]
        )
        return response.content[0].text

class ChatClientFactory:
    @staticmethod
    def create_client(provider: str, api_key: str, 
                      model: Optional[str] = None,
                      system_msg: Optional[str] = None) -> ChatClient:
        if system_msg is None:
            system_msg = "You are a helpful AI assistant."
        if provider.lower() == "openai":
            return OpenAIChatClient(api_key, model or "gpt-4")
        elif provider.lower() == "anthropic":
            return AnthropicChatClient(api_key, 
                                       model or "claude-3-sonnet-20240229",
                                       system_msg)
        else:
            raise ValueError(f"Unsupported provider: {provider}")


if __name__ == "__main__":

    api_key = os.environ.get("ANTHROPIC")
    client = ChatClientFactory.create_client("anthropic", api_key)
    response = client.chat("Hello, how are you?")
    print(response)