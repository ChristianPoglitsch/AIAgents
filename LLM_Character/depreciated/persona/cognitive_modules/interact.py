import random
from typing import Tuple

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessages
from LLM_Character.persona.cognitive_modules.interacting.chatting import chat_react
from LLM_Character.persona.cognitive_modules.interacting.reacting import should_react
from LLM_Character.persona.cognitive_modules.interacting.waiting import wait_react
from LLM_Character.persona.cognitive_modules.retrieve import EventContext
from LLM_Character.persona.memory_structures.associative_memory.\
    associative_memory import (
        AssociativeMemory,
    )
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)


def interact(
    scratch: PersonaScratch,
    mem: AssociativeMemory,
    personas: dict[str, Tuple[PersonaScratch, AssociativeMemory]],
    retrieved: dict[str, EventContext],
    model: LLM_API,
) -> str:
    focused_event = False
    if retrieved.keys():
        focused_event = choose_retrieved(scratch, retrieved)

    if focused_event:
        reaction_mode = should_react(
            scratch, mem, personas, focused_event, model)
        if reaction_mode:
            if reaction_mode[:9] == "chat with":
                target_scratch, target_mem = personas[reaction_mode[9:].strip(
                )]
                chat_react(scratch, mem, target_scratch, target_mem, personas)
            elif reaction_mode[:4] == "wait":
                wait_react(scratch, reaction_mode)

    if scratch.act_event[1] != "chat with":
        scratch.chatting_with = None
        scratch.chat = AIMessages()
        scratch.chatting_end_time = None

    curr_persona_chat_buffer = scratch.chatting_with_buffer
    for persona_name, _ in curr_persona_chat_buffer.items():
        if persona_name != scratch.chatting_with:
            scratch.chatting_with_buffer[persona_name] -= 1

    return scratch.act_address


def choose_retrieved(
    cscratch: PersonaScratch, retrieved: dict[str, EventContext]
) -> EventContext:
    # dont think we need this, since self events are not sent?
    # but still could be left here:
    copy_retrieved = retrieved.copy()
    for event_desc, rel_ctx in copy_retrieved.items():
        curr_event = rel_ctx.curr_event
        if curr_event.subject == cscratch.name:
            del retrieved[event_desc]

    # Always choose persona first.
    priority = []
    for _, rel_ctx in retrieved.items():
        curr_event = rel_ctx.curr_event
        if ":" not in curr_event.subject and curr_event.subject != cscratch.name:
            priority += [rel_ctx]
    if priority:
        return random.choice(priority)

    # Skip idle.
    for event_desc, rel_ctx in retrieved.items():
        curr_event = rel_ctx.curr_event
        if "is idle" not in event_desc:
            priority += [rel_ctx]
    if priority:
        return random.choice(priority)
    return None
