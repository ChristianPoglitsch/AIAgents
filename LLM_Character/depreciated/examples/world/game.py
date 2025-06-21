"""
This script demonstrates an end-to-end example of
interacting with the `ReverieServer`,
The script showcases how to load the server,
and send messages.
"""

import logging

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_local import LocalComms
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.persona.persona import Persona
from LLM_Character.persona.user import User
from LLM_Character.util import BASE_DIR, LOGGER_NAME, setup_logging
from LLM_Character.world.game import ReverieServer

if __name__ == "__main__":
    setup_logging("examples_main")
    logger = logging.getLogger(LOGGER_NAME)

    florian = Persona("Florian")
    florian.load_from_file(
        BASE_DIR + "/LLM_Character/storage/localhost/default/personas/Florian"
    )

    user = User("Louis")

    modelb = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelb.init(model_id)

    modela = OpenAIComms()
    model_id = "gpt-4"
    modela.init(model_id)

    for model in modela, modelb:
        wrapper_model = LLM_API(model)
        message = "hi"

        r = ReverieServer("sim_code", None, "notlocalhost")

        a = r.is_loaded()
        out1 = r.prompt_processor(
            user.scratch.name, florian.scratch.name, message, wrapper_model
        )
        assert a is False
        assert out1 is None

        logger.info(out1)

        r.start_processor()

        a = r.is_loaded()
        out1 = r.prompt_processor(
            user.scratch.name, florian.scratch.name, message, wrapper_model
        )
        assert a
        assert out1 is not None

        logger.info(out1)
