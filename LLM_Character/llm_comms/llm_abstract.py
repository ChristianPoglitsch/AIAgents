from abc import ABC, abstractmethod

from LLM_Character.messages_dataclass import AIMessages


class LLMComms(ABC):
    @abstractmethod
    def init():
        pass

    @abstractmethod
    def send_text(self, prompt: AIMessages) -> str:
        pass

    @abstractmethod
    def send_embedding(text: str):
        pass

    @abstractmethod
    def set_params():
        pass

    @abstractmethod
    def get_model_name():
        pass
    
    @abstractmethod
    def set_temperature():
        pass  
