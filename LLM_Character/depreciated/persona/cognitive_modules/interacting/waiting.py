import datetime

from LLM_Character.persona.cognitive_modules.conversing.ending import _create_react
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)


def wait_react(scratch: PersonaScratch, reaction_mode: str):
    inserted_act = f'waiting to start {scratch.act_description.split("(")[-1][:-1]}'
    end_time = datetime.datetime.strptime(
        reaction_mode[6:].strip(), "%B %d, %Y, %H:%M:%S"
    )
    inserted_act_dur = (
        (end_time.minute + end_time.hour * 60)
        - (scratch.curr_time.minute + scratch.curr_time.hour * 60)
        + 1
    )

    act_address = f"<waiting> {scratch.curr_tile[0]} {scratch.curr_tile[1]}"
    act_event = (
        scratch.name,
        "waiting to start",
        scratch.act_description.split("(")[-1][:-1],
    )
    chatting_with = None
    chat = None
    chatting_with_buffer = None
    chatting_end_time = None

    act_obj_description = None
    act_obj_event = (None, None, None)

    _create_react(
        scratch,
        inserted_act,
        inserted_act_dur,
        act_address,
        act_event,
        chatting_with,
        chat,
        chatting_with_buffer,
        chatting_end_time,
        act_obj_description,
        act_obj_event,
    )
