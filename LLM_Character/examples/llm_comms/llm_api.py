"""
This script showcases how to use the LLM_API interface with different language models.
It shows how to call the functions `query_text` and `get_embedding`
"""

import logging

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_local import LocalComms
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.util import LOGGER_NAME, setup_logging

if __name__ == "__main__":
    setup_logging("examples_llm_api")
    logger = logging.getLogger(LOGGER_NAME)

    x = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    x.init(model_id)

    y = OpenAIComms()
    model_id = "gpt-4"
    y.init(model_id)
    modelb = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelb.init(model_id)

    modela = OpenAIComms()
    model_id = "gpt-4"
    modela.init(model_id)

    for model in modela, modelb:
        hf = LLM_API(model)

        messages = AIMessages()
        m1 = AIMessage("Jarek", "Hello, how are you?", "user", "MessageAI")
        m2 = AIMessage(
            "Camila",
            "I'm fine, thank you! How can I assist you today?",
            "assistant",
            "MessageAI",
        )
        m3 = AIMessage("Jarek", "Can you tell me a joke?", "user", "MessageAI")
        m4 = AIMessage(
            "Camila",
            "Why don't scientists trust atoms?\
                    Because they make up everything!",
            "assistant",
            "MessageAI",
        )
        messages.add_message(m1)
        messages.add_message(m2)
        messages.add_message(m3)
        messages.add_message(m4)

        response = hf.query_text(messages)

        logger.info("Model response:")
        logger.info(response)

        text = "Lorem upsum"
        embedding = hf.get_embedding(text)

        logger.info("Model embedding of text:")
        logger.info(embedding)
