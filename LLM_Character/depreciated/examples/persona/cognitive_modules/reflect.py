"""
This script demonstrates how to use both `chatting` and `reflect` functions.
"""

import datetime
import logging

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_local import LocalComms
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.persona.cognitive_modules.converse import chatting
from LLM_Character.persona.cognitive_modules.reflect import reflect
from LLM_Character.persona.persona import Persona
from LLM_Character.persona.user import User
from LLM_Character.util import BASE_DIR, LOGGER_NAME, setup_logging

if __name__ == "__main__":
    setup_logging("examples_reflect")
    logger = logging.getLogger(LOGGER_NAME)

    logger.info("starting take off ...")

    modelb = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelb.init(model_id)

    modela = OpenAIComms()
    model_id = "gpt-4"
    modela.init(model_id)
    curr_time = datetime.datetime(2024, 9, 4, 9, 23, 42)

    for model in modela, modelb:
        wrapper_model = LLM_API(model)

        person = Persona("Camila")
        person.load_from_file(
            BASE_DIR + "/LLM_Character/storage/localhost/default/personas/Camila"
        )
        person.scratch.curr_time = curr_time

        user = User("Louis")

        message = "bye bye see you tomorrow, end this conversation"
        response = chatting(
            user.scratch, person.scratch, person.a_mem, message, wrapper_model
        )

        reflect(person.scratch, person.a_mem, wrapper_model)
