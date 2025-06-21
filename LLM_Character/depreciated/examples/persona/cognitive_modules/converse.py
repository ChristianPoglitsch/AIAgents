"""
This script demonstrates how to use the `chatting` functionality
to interact with a persona.
The script loads two different language models (OpenAI's GPT-4 and a local model),
initializes them, and runs a series of chat interactions with a predefined persona.
"""

import datetime
import logging

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_local import LocalComms
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.persona.cognitive_modules.converse import chatting
from LLM_Character.persona.persona import Persona
from LLM_Character.persona.user import User
from LLM_Character.util import BASE_DIR, LOGGER_NAME, setup_logging

if __name__ == "__main__":
    setup_logging("examples_converse")
    logger = logging.getLogger(LOGGER_NAME)

    modela = OpenAIComms()
    model_id1 = "gpt-4"
    modela.init(model_id1)

    modelb = LocalComms()
    model_id2 = "mistralai/Mistral-7B-Instruct-v0.2"
    modelb.init(model_id2)

    models = [modela, modelb]
    for model in models:
        # you have to reload persona, since you cannot have one
        # chat or a part of the chat being produced by
        # one type of LLM and the other by another etc.
        # it always needs to be he same LLM.

        person = Persona("Florian")
        person.load_from_file(
            BASE_DIR + "/LLM_Character/storage/localhost/default/personas/Florian"
        )
        person.scratch.curr_time = datetime.datetime(21, 3, 4)
        user = User("Louis")

        wrapped_model = LLM_API(model)
        logger.info(model.__class__.__name__)

        # -----------------------------------------------------------------------------

        message = "But alors, you are French?"
        response, emotion, trust, end = chatting(
            user.scratch, person.scratch, person.a_mem, message, wrapped_model
        )
        assert isinstance(response, str)
        assert emotion in [
            "neutral",
            "happy",
            "angry",
            "disgust",
            "fear",
            "surprised",
            "sad",
        ]
        assert isinstance(trust, int)
        assert 0 <= trust <= 10
        assert isinstance(end, bool)

        logger.info("data: ")
        logger.info(f"{response} | Emotion: {emotion}, Trust: {trust}, End: {end}")

        message = "Das ist eine kolossale Konspiration !"
        response, emotion, trust, end = chatting(
            user.scratch, person.scratch, person.a_mem, message, wrapped_model
        )
        assert isinstance(response, str)
        assert emotion in [
            "neutral",
            "happy",
            "angry",
            "disgust",
            "fear",
            "surprised",
            "sad",
        ]
        assert isinstance(trust, int)
        assert 0 <= trust <= 10
        assert isinstance(end, bool)

        logger.info("data: ")
        logger.info(f"{response} | Emotion: {emotion}, Trust: {trust}, End: {end}")

        message = "bye, see you later"
        response, emotion, trust, end = chatting(
            user.scratch, person.scratch, person.a_mem, message, wrapped_model
        )
        assert isinstance(response, str)
        assert emotion in [
            "neutral",
            "happy",
            "angry",
            "disgust",
            "fear",
            "surprised",
            "sad",
        ]
        assert isinstance(trust, int)
        assert 0 <= trust <= 10
        assert isinstance(end, bool)

        logger.info("data: ")
        logger.info(f"{response} | Emotion: {emotion}, Trust: {trust}, End: {end}")
