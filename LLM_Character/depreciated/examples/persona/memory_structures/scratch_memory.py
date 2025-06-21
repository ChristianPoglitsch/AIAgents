"""
This script demonstrates the usage of the `PersonaScratch` class.
The script illustrates how to create a persona's scratch memory,
add new actions, check the status of these actions, update persona data,
and finally save and load the persona's scratch memory to and from a file.
"""

import datetime
import logging

from LLM_Character.communication.incoming_messages import (
    OneLocationData,
    PersonaScratchData,
)
from LLM_Character.messages_dataclass import AIMessages
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (  # noqa: E501
    PersonaScratch,
)
from LLM_Character.util import BASE_DIR, LOGGER_NAME, setup_logging

if __name__ == "__main__":
    setup_logging("examples_associative_memory")
    logger = logging.getLogger(LOGGER_NAME)

    # ---------------------------------

    scratch = PersonaScratch(name="A(l)di")
    scratch.curr_time = datetime.datetime.now()

    scratch.add_new_action(
        action_address="123 Main St",
        action_duration=60,
        action_description="Meeting with team",
        action_event=("Meeting", "Team", "Discuss project"),
        chatting_with="Alice",
        chat=AIMessages(),
        chatting_with_buffer={"Alice": "Discuss project details"},
        chatting_end_time=datetime.datetime.now() + datetime.timedelta(minutes=60),
    )

    logger.info(scratch.act_check_finished())

    logger.info(scratch.get_f_daily_schedule_index())

    logger.info(scratch.get_curr_event_and_desc())

    logger.info(scratch.get_str_daily_schedule_summary())

    scratch.update(
        PersonaScratchData(
            curr_location=OneLocationData(
                world="Graz", sector="Sure", arena="Dont know"
            ),
            first_name="John",
            last_name="Doe",
            age=30,
            innate="Brave",
            learned="Strategic thinking",
            currently="Project Manager",
            lifestyle="Active",
            look="Casual",
            living_area=OneLocationData(
                world="Graz", sector="your moms house", arena="bedroom :)"
            ),
            recency_w=9,
            relevance_w=8,
            importance_w=1,
            recency_decay=0.98,
            importance_trigger_max=160,
            importance_trigger_curr=150,
            importance_ele_n=10,
        )
    )

    scratch.save(
        BASE_DIR
        + "/LLM_Character/examples/persona/memory_structures/temp/Florian/scratch.json"
    )
    scratch.load_from_file(
        BASE_DIR + "/LLM_Character/storage/localhost/default/personas/Florian/"
    )

    logger.info(scratch.get_info())
    logger.info(scratch.act_summary_str())
