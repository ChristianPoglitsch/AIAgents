import logging
import os
import warnings
from typing import List, Optional, Union

import torch
from openai.types import Embedding
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from LLM_Character.finetuning.models import load_base_model
from LLM_Character.llm_comms.llm_abstract import LLMComms
from LLM_Character.messages_dataclass import AIMessage, AIMessages

# in order to prevent the terminal to be cluttered from all the
# torch/transformers warnings.
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


class LocalComms(LLMComms):
    """
    class responsible for sending messages to LocalLLM's.
    In short, a wrapper class for HuggingFace API.
    """

    def __init__(self):
        self._model: PreTrainedModel = None
        self._tokenizer: PreTrainedTokenizer = None
        self._embedding_model: SentenceTransformer = None
        self._model_name = ''

        self.max_tokens = 100
        self.temperature = 0.8
        self.top_p = 1        

    def init(self, base_model_id: str, finetuned_model_id: str = None):
        """
        Initialize the model and tokenizer using the specified model ID.
        If the model ID refers to a local fine tuned model,
        then it will prioritize that over a model on huggingFace.
        """

        # FIXME: check if the model id's are valid before laoding them.
        # or use try catch when loading them.
        # especially for semantic comparason,
        # you have to check if the model is made for that.
        # the same can be done for chatting, checking if the model is a text
        # generator type.

        if finetuned_model_id:
            self._model, self._tokenizer = self._load_model_loc(
                base_model_id, finetuned_model_id
            )
        else:
            self._model, self._tokenizer = self._load_model_hf(base_model_id)

        # FIXME: cannot use mistral, since it is not an embedding model.
        self._embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._model_name = base_model_id

    def get_model_name(self) -> str:
        """Get model name/id

        Returns:
            str: model name
        """
        return self._model_name

    def send_text(self, prompt: AIMessages, max_length=100) -> Optional[str]:
        """
        send a prompt to local endpoint for chat completions.
        """
        if len(prompt) == 0:
            return None

        response = self._request(self._model, self._tokenizer, prompt, max_length)
        response_decoded = self._decode_request(response)

        return response_decoded

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

    # FIXME: change parameters to dictionary via args or kwargs or whatever.
    # there will be parameters present in a certain
    # model that will not be in another model etc.
    # see huggingface.
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

    # FIXME: should be changed, validate the parameters
    # via Huggingface API of the model.
    # certain models will have diferent parameters limits.
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

    def _check_local(self, finetuned_model_id):
        return os.path.isdir(f"{finetuned_model_id}")

    def _load_model_loc(
        self, model_id: str, adapters_id: str
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        base_model = load_base_model(model_id)
        model = PeftModel.from_pretrained(base_model, adapters_id)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="right", use_fast=False
        )
        return model, tokenizer

    def _load_model_hf(
        self, model_id: str
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load the model and tokenizer from the specified model ID.

        Returns:
            tuple: A tuple containing the model and tokenizer.
        """

        # The `load_in_4bit` and `load_in_8bit` arguments
        # are deprecated and will be removed in the future versions.
        # Please, pass a `BitsAndBytesConfig` object in `quantization_config`
        # argument instead.
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(  # device_map="auto"
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16, # token=' '
        )

        model.config.sliding_window = 4096
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # tokenizer.bos_token = "<bos>"
        # tokenizer.pad_token = "<pad>"
        tokenizer.cls_token = "<cls>"
        tokenizer.sep_token = "<s>"
        tokenizer.mask_token = "<mask>"
        return model, tokenizer

    def _request(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        message: Union[AIMessages, AIMessage],
        max_length: int,
    ) -> str:
        """
        Generate a response from the model based on the input messages.

        Args:
            model: The model to use for querying.
            tokenizer: The tokenizer to use for querying.
            messages: The messages to query the model with.

        Returns:
            str: The model's response.
        """
        # start_time = time.process_time()

        device = "cuda"
        inputs = tokenizer.apply_chat_template(
            message.get_formatted(), return_tensors="pt"
        ).to(device)  # tokenize=False)

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.2,  # 1.0
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_length,
        )
        generation_config.eos_token_id = tokenizer.eos_token_id

        #outputs = model.generate(inputs, generation_config=generation_config)
        outputs = model.generate(input_ids=inputs["input_ids"], generation_config=generation_config)
        #response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Temp for Mistral 7 v03
        prompt_length = inputs.shape[1]
        response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

        # print("Processing time: " + str(time.process_time() - start_time) + " sec")
        return response

    def _decode_request(self, message: str) -> str:
        """
        Decode the response message from the model.

        Example:
            Given a raw response message:
            "[INST] nice [/INST] indeed [INST] nice [/INST] indeed [INST] nice [/INST]n"
            It will return:
            "n"
        """
        response = message.replace("[/INST]", "[INST]").split("[INST]")
        return response[len(response) - 1]

    def _requese_emb(self, keywords: str) -> Tensor:
        embeddings = self._embedding_model.encode(keywords, convert_to_tensor=True)
        return embeddings.cpu().tolist()
