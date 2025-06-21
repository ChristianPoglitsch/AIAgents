"""
This script demonstrates how to use the retrieval functionalities,
specifically the `retrieve_focal_points` and `retrieve_contextual_events` methods.
These methods are part of the cognitive modules for
querying associative memory and handling perceived events.
"""

import datetime
import logging

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_local import LocalComms
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.persona.cognitive_modules.retrieve import (
    retrieve_contextual_events,
    retrieve_focal_points,
)
from LLM_Character.persona.memory_structures.associative_memory.concept_node import (
    ConceptNode,
)
from LLM_Character.persona.persona import Persona
from LLM_Character.util import BASE_DIR, LOGGER_NAME, setup_logging

if __name__ == "__main__":
    setup_logging("examples_retrieve")
    logger = logging.getLogger(LOGGER_NAME)

    modela = OpenAIComms()
    model_id1 = "gpt-4"
    modela.init(model_id1)

    modelb = LocalComms()
    model_id2 = "mistralai/Mistral-7B-Instruct-v0.2"
    modelb.init(model_id2)

    models = [modela, modelb]
    for model in models:
        logger.info(model.__class__.__name__)

        wrapped_model = LLM_API(model)

        person = Persona("Camila")
        person.load_from_file(
            BASE_DIR + "/LLM_Character/storage/localhost/default/personas/Camila"
        )

        text = "Frederiek went to the shop"
        created = datetime.datetime(21, 3, 4)
        person.scratch.curr_time = created

        expiration = None
        s = "Lorem"
        p = "went"
        o = "shop"
        description1 = "frederiek went to the shop"
        description2 = "frederiek went to the cornershop"
        keywords = ["shop"]
        poignancy = 3
        embedding_pair = [description1, wrapped_model.get_embedding(description1)]
        filling = None

        node1 = person.a_mem.add_thought(
            created,
            expiration,
            s,
            p,
            o,
            description1,
            keywords,
            poignancy,
            embedding_pair,
            filling,
        )
        node2 = person.a_mem.add_event(
            created,
            expiration,
            s,
            p,
            o,
            description2,
            keywords,
            poignancy,
            embedding_pair,
            filling,
        )

        # -------------------------------------------------------------
        statement = "frederiek went shopping?"
        retrieved1 = retrieve_focal_points(
            person.scratch, person.a_mem, [statement], wrapped_model
        )

        length = len(retrieved1[statement])
        logger.info(
            f"length of retrieved nodes for the following statement: {statement}"
        )

        # because there are two events in memory (see above add_thought and add_event)
        # that are related to the statement "frederiek went shopping"
        # so two events should be retrieved from memory.
        assert length == 2

        logger.info(length)
        for i in range(length):
            logging.info(retrieved1[statement][i].description)

        # -------------------------------------------------------------

        perceived_node = ConceptNode(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            s="shop",
            p=None,
            o=None,
            description=None,
            embedding_key=None,
            poignancy=None,
            keywords=None,
            filling=None,
        )
        retrieved3 = retrieve_contextual_events(person.a_mem, [perceived_node])

        logger.info(
            "output from the retrieved_contextual_events \
                method from the two perceived nodes above. \
                the question is, which nodes in memory are \
                related to the perceived nodes node1, node2. \
                obviously the same nodes as well. "
        )

        assert len(retrieved3.items()) == 1
        for _, j in retrieved3.items():
            assert len(j.events) == 1
            assert len(j.thoughts) == 1
