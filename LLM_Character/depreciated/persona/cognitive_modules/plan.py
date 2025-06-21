"""
Plan new steps or quests.
"""

import logging

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.persona.cognitive_modules.planning.determine_action import (
    _determine_action,
)
from LLM_Character.persona.cognitive_modules.planning.long_term_planning import (
    _long_term_planning,
)
from LLM_Character.persona.memory_structures.associative_memory.associative_memory import (  # noqa: E501
    AssociativeMemory,
)
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.memory_structures.spatial_memory import MemoryTree
from LLM_Character.util import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def plan(
    scratch: PersonaScratch,
    a_mem: AssociativeMemory,
    s_mem: MemoryTree,
    new_day: str,
    model: LLM_API,
):
    if new_day:
        logger.info("Planning for long term")
        _long_term_planning(scratch, a_mem, new_day, model)

    if scratch.act_check_finished():
        logger.info("Determining Action")
        _determine_action(scratch, s_mem, model)

    return scratch.act_address
