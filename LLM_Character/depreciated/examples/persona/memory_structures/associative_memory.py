"""
This script demonstrates the use of the `AssociativeMemory` class.
The script showcases various operations including adding
thoughts, chats, and events to the memory, as well as retrieving
and summarizing relevant memories. Additionally, it illustrates
how to load and save the associative memory to and from a file.
"""

import datetime
import logging

from LLM_Character.persona.memory_structures.associative_memory.associative_memory import (  # noqa: E501
    AssociativeMemory,
)
from LLM_Character.util import BASE_DIR, LOGGER_NAME, setup_logging

if __name__ == "__main__":
    setup_logging("examples_associative_memory")
    logger = logging.getLogger(LOGGER_NAME)

    # ---------------------------------

    memory = AssociativeMemory()

    created = datetime.datetime.now()
    expiration = None
    s = "Thought Subject"
    p = "Thought Predicate"
    o = "Thought Object"
    description = "A description of the thought."
    keywords = ["important", "urgent"]
    poignancy = 7
    embedding_pair = ("embedding_key_1", [0.1, 0.2, 0.3])
    filling = []

    node = memory.add_thought(
        created,
        expiration,
        s,
        p,
        o,
        description,
        keywords,
        poignancy,
        embedding_pair,
        filling,
    )

    logger.info(node)

    # ---------------------------------

    memory = AssociativeMemory()

    created = datetime.datetime.now()
    expiration = None
    s = "Chat Subject"
    p = "Chat Predicate"
    o = "Chat Object"
    description = "A description of the chat."
    keywords = ["chat", "conversation"]
    poignancy = 6
    embedding_pair = ("embedding_key_2", [0.4, 0.5, 0.6])
    filling = [["user", "Hi!"], ["bot", "Hello!"]]

    node = memory.add_chat(
        created,
        expiration,
        s,
        p,
        o,
        description,
        keywords,
        poignancy,
        embedding_pair,
        filling,
    )

    logger.info(node)

    # ---------------------------------

    memory = AssociativeMemory()

    created = datetime.datetime.now()
    expiration = None
    s = "Event Subject"
    p = "Event Predicate"
    o = "Event Object"
    description = "A description of the event."
    keywords = ["event", "important"]
    poignancy = 7
    embedding_pair = ("embedding_key_3", [0.7, 0.8, 0.9])
    filling = []

    node = memory.add_event(
        created,
        expiration,
        s,
        p,
        o,
        description,
        keywords,
        poignancy,
        embedding_pair,
        filling,
    )

    logger.info(node)

    # ---------------------------------

    relevant_thoughts = memory.retrieve_relevant_thoughts(
        s_content="Thought Subject",
        p_content="Thought Predicate",
        o_content="Thought Object",
    )

    logger.info(relevant_thoughts)

    relevant_events = memory.retrieve_relevant_events(
        s_content="Event Subject", p_content="Event Predicate", o_content="Event Object"
    )

    logger.info(relevant_events)

    logger.info(memory.get_str_seq_thoughts())
    logger.info(memory.get_str_seq_chats())
    last_chat = memory.get_last_chat(target_persona_name="user")
    logger.info(last_chat)
    last_chat = memory.get_last_chat(target_persona_name="bot")
    logger.info(last_chat)
    summarized_events = memory.get_summarized_latest_events(retention=5)
    logger.info(summarized_events)
    summarized_events = memory.get_summarized_latest_events(retention=5)
    logger.info(summarized_events)

    # ---------------------------------

    memory = AssociativeMemory()
    memory.load_from_file(
        BASE_DIR
        + "/LLM_Character/storage/localhost/default/personas/Florian/associative_memory"
    )

    node = memory.add_thought(
        created,
        expiration,
        s,
        p,
        o,
        description,
        keywords,
        poignancy,
        embedding_pair,
        filling,
    )
    node = memory.add_event(
        created,
        expiration,
        s,
        p,
        o,
        description,
        keywords,
        poignancy,
        embedding_pair,
        filling,
    )
    node = memory.add_chat(
        created,
        expiration,
        s,
        p,
        o,
        description,
        keywords,
        poignancy,
        embedding_pair,
        filling,
    )
    memory.save(
        BASE_DIR + "/LLM_Character/examples/persona/memory_structures/temp/Florian/"
    )
