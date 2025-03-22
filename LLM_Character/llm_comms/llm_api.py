from xmlrpc.client import boolean
from LLM_Character.llm_comms.llm_abstract import LLMComms
from LLM_Character.messages_dataclass import AIMessages


class LLM_API:
    """
    A class to handle operations related to LLM models,
    including loading models, querying models
    """

    def __init__(self, LLMComms: LLMComms, debug : boolean = False):
        self._model = LLMComms
        self._model_name = LLMComms.get_model_name()
        self._debug = debug

    def query_text(self, message: AIMessages) -> str:
        if self._debug:
            print('--- --- ---')
            print(message.get_formatted())
            print('--- --- ---')
        result = self._model.send_text(message, self._model.max_tokens)
        if self._debug:
            print('*** *** ***')
            print(result)
            print('*** *** ***')
        return result

    def get_embedding(self, text: str) -> list[float]:
        """
        retrieves the text embedding.
        """
        return self._model.send_embedding(text)

    def get_model_name(self) -> str:
        return self._model.get_model_name()
    
    def set_max_tokens(self, value: int):
        self._model.max_tokens = value
        
    def set_temperature(self, temperature: float):
        self._model.set_temperature(temperature)
