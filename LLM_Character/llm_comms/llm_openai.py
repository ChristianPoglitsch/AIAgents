"""inspired by

https://github.com/adimis-ai/Large-Language-Model-LLM-Wrapper-from-Scratch-using-Openai-models/blob/main/Large_Language_Model_(LLM)_Wrapper_from_Scratch_using_Openai_models.ipynb
https://github.com/openai/openai-python
https://github.com/joonspk-research/generative_agents

"""

import logging
import warnings
from typing import List, Optional

from openai import OpenAI

# tiktoken is a fast BPE tokeniser for use with OpenAI's models.
from openai.types import ChatModel, CreateEmbeddingResponse, Embedding

from LLM_Character.llm_comms.llm_abstract import LLMComms
from LLM_Character.messages_dataclass import AIMessages
from LLM_Character.util import API_KEY, LOGGER_NAME

# in order to prevent the terminal to be cluttered from all the
# torch/transformers warnings.
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger(LOGGER_NAME)


class OpenAIComms(LLMComms):
    """class responsible for sending messages to openAI API"""

    def __init__(self):
        self.client = None
        self.model_name = None

        self.max_tokens = 100
        self.temperature = 0.8
        self.top_p = 1
        self.n = 1
        self.max_retry_attempts = 3
        self.presence_penalty = None
        self.frequency_penalty = None
        # "stop": ["\n"]

    def init(self, model: str):
        """
        if model name is not valid, it raises a value exception error.
        """
        if not self._check_valid_model_chat(model):
            raise ValueError(
                "The model type does not exist or is not \
                camptabible for chat completion."
            )

        self.model_name = model
        self.client = OpenAI(api_key=API_KEY)

    def get_model_name(self) -> str:
        """Get model name/id

        Returns:
            str: model name
        """
        return self.model_name

    # FIXME: use dict as arguments for gpt model instead of positional
    # arguments.
    def send_text(self, aimessages: AIMessages, max_length=100) -> Optional[str]:
        """send a prompt to openAI endpoint for chat completions.

        Args:
            prompt (AIMessages): _description_

        Returns:
            _type_: _description_
        """
        if len(aimessages) == 0:
            return None

        prompt = aimessages.get_formatted()

        return self._request(prompt)

    def send_embedding(self, keywords: str) -> Optional[List[Embedding]]:
        """send a prompt to openAI endpoint for embeddings.

        Args:
            keywords (str): keywords for which you convert it to embeddings.

        Returns:
            List[float]: The embedding vector, which is a list of floats.
            The length of vector depends on the model as listed in the
            [embedding guide](https://platform.openai.com/docs/guides/embeddings).
        """
        if len(keywords) == 0:
            return None
        return self._requese_emb(keywords)

    def set_params(
        self,
        max_tokens: int,
        temperature: int,
        top_p: int,
        amount_responses: int,
        max_retry_attemps: int,
        presence_penalty=None,
        frequency_penalty=None,
    ):
        if not self._validate_inputs(
            max_tokens,
            temperature,
            top_p,
            amount_responses,
            max_retry_attemps,
            presence_penalty,
            frequency_penalty,
        ):
            return None

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = amount_responses
        self.max_retry_attempts = max_retry_attemps
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    # ---------
    #  PRIVATE
    # ---------

    def _validate_inputs(
        self,
        max_tokens,
        temperature,
        top_p,
        n,
        max_retry_attemps,
        presence_penalty,
        frequency_penalty,
    ):
        b1 = max_tokens > 0
        b2 = temperature >= 0 and temperature <= 2
        b3 = top_p >= 0 and top_p <= 1
        b4 = n >= 1 and n <= 128
        b5 = max_retry_attemps >= 0
        b6 = presence_penalty or (presence_penalty >= -2 and presence_penalty <= 2)
        b7 = frequency_penalty or (frequency_penalty >= -2 and frequency_penalty <= 2)

        if all[b1, b2, b3, b4, b5, b6, b7]:
            return True
        return False

    def _request(self, messages: List[dict]) -> Optional[str]:
        """
        roles of the messages should be "system", "user", "assistant".

        Args:
            messages (AIMessages): the messages for chat completion.

        Returns:
            Optional[str]: the completion of the provided chat.
        """
        try:
            response = self.client.with_options(
                max_retries=self.max_retry_attempts
            ).chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=self.max_tokens,
                n=self.n,
                # FIXME: this could be very usefull for us, since we do use json object
                # in which we want the reponse to be in, and the trust level, etc...
                # response_format= ? "Must be one of `text` or `json_object`."
                # bv. response_format={"type": "json_object"}
                # temperature: float | NotGiven | None = NOT_GIVEN,
                # top_p: float | NotGiven | None = NOT_GIVEN,
                # frequency_penalty
                # presence_penalty
                # stop
            )
        except Exception as e:
            logger.error("openAI request failed")
            logger.error(e)
            return None

        # see openai.yaml under
        # CreateChatCompletionResponse schema
        # ChatCompletionResponseMessage schema
        # also see python_client.md

        # choose first message option if there are multiple options, see n
        # parameter
        return response.choices[0].message.content

    def _requese_emb(
        self, keywords: str, model_name="text-embedding-3-small"
    ) -> Optional[List[Embedding]]:
        try:
            response: CreateEmbeddingResponse = self.client.embeddings.create(
                model=model_name, input=keywords
            )

        except Exception as e:
            logger.error("openAI request failed")
            logger.error(e)
            return None

        embedding = response.data[0].embedding
        return embedding

    def _check_valid_model_chat(self, model: str):
        """
            check if the model type does exist.
            And if it exists, check if it compatible for chat completion.
        Args:
            model (str): model id
        """

        return model in str(ChatModel)

    def _check_valid_model_emb(self, model: str):
        """
            check if the model type does exist.
            And if it exists, check if it compatible for chat completion.
        Args:
            model (str): model id
        """

        return model in [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]
