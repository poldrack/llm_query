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
                 system_msg: str = None, max_tokens: int = 16384):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.system_msg = system_msg or "You are a helpful AI assistant."
        
    def chat(self, message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content
    
    # ala https://platform.openai.com/docs/guides/batch
    def create_batch_request(self, id, message: str) -> str:
        return {"custom_id": id, 
         "method": "POST", 
         "url": "/v1/chat/completions",
         "body": {"model": self.model, 
                 "messages": [{"role": "system", "content": self.system_msg},
                              {"role": "user", "content": message}],
                 "max_tokens": self.max_tokens}}


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
                      system_msg: Optional[str] = None,
                      **kwargs) -> ChatClient:
        if system_msg is None:
            system_msg = "You are a helpful AI assistant."
        if provider.lower() == "openai":
            return OpenAIChatClient(api_key, model or "gpt-4", system_msg, **kwargs)
        elif provider.lower() == "anthropic":
            return AnthropicChatClient(api_key, 
                                     model or "claude-3-sonnet-20240229",
                                     system_msg,
                                     **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

if __name__ == "__main__":

    api_key = os.environ.get("ANTHROPIC")
    client = ChatClientFactory.create_client("anthropic", api_key)
    response = client.chat("Hello, how are you?")
    print(response)