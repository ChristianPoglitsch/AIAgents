"""
This script demonstrates how to use the `perceive` function.
It involves loading a persona, setting its context,
and then using different LLM models to process perceived
events by the persona.
"""

import datetime
import logging

from LLM_Character.communication.incoming_messages import EventData, OneLocationData
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_local import LocalComms
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.persona.cognitive_modules.perceive import perceive
from LLM_Character.persona.persona import Persona
from LLM_Character.persona.user import User
from LLM_Character.util import BASE_DIR, LOGGER_NAME, setup_logging

if __name__ == "__main__":
    setup_logging("examples_perceive")
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

        location_data = OneLocationData(
            world="Eryndor",
            sector="The Shattered Isles",
            arena="Crimson Hollow Tavern",
        )
        s = "Eryndor:The Shattered Isles:Crimson Hollow Tavern:Ancient Crystal"
        perceived_events = [
            EventData(
                action_event_subject=s,
                action_event_predicate=None,
                action_event_object=None,
                action_event_description=None,
            ),
            EventData(
                action_event_subject="Camila",
                action_event_predicate=None,
                action_event_object=None,
                action_event_description=None,
            ),
        ]

        events = perceive(
            person.scratch,
            person.a_mem,
            person.s_mem,
            location_data,
            perceived_events,
            wrapped_model,
        )

        assert isinstance(events, list)
        assert len(events) == 2
        for event in events:
            logger.info(event)
