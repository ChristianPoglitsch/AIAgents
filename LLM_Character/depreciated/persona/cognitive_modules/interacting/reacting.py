import datetime

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.persona.cognitive_modules.retrieve import EventContext
from LLM_Character.persona.memory_structures.associative_memory.associative_memory import (
    AssociativeMemory,
)
from LLM_Character.persona.memory_structures.associative_memory.concept_node import (
    ConceptNode,
)
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.interact_prompts.decide_to_react import (
    run_prompt_decide_to_react,
)
from LLM_Character.persona.prompt_modules.interact_prompts.decide_to_talk import (
    run_prompt_decide_to_talk,
)


def should_react(
    persona_scratch: PersonaScratch,
    persona_amem: AssociativeMemory,
    personas: dict[str, PersonaScratch],
    focused_event: EventContext,
    model: LLM_API,
):
    # If the persona is chatting right now, default to no reaction
    if persona_scratch.chatting_with:
        return False
    if "<waiting>" in persona_scratch.act_address:  # FIXME: where added in the project?
        return False

    curr_event = focused_event.curr_event
    # this is a persona event if
    if ":" not in curr_event.subject:
        tscratch, _ = personas[curr_event.subject]
        if lets_talk(persona_scratch, persona_amem, tscratch, focused_event, model):
            return f"chat with {curr_event.subject}"

        react_mode = lets_react(persona_scratch, tscratch, focused_event, model)
        return react_mode

    return False


def lets_talk(
    init_scratch: PersonaScratch,
    init_amem: AssociativeMemory,
    target_scratch: PersonaScratch,
    retrieved: EventContext,
    model: LLM_API,
) -> bool:
    # you cant talk if you were doing something.
    if (
        not target_scratch.act_address
        or not target_scratch.act_description
        or not init_scratch.act_address
        or not init_scratch.act_description
    ):
        return False

    if (
        "sleeping" in target_scratch.act_description
        or "sleeping" in init_scratch.act_description
    ):
        return False

    if init_scratch.curr_time.hour == 23:
        return False

    if "<waiting>" in target_scratch.act_address:
        return False

    if target_scratch.chatting_with or init_scratch.chatting_with:
        return False

    if target_scratch.name in init_scratch.chatting_with_buffer:
        if init_scratch.chatting_with_buffer[target_scratch.name] > 0:
            return False

    if generate_decide_to_talk(
        init_scratch, init_amem, target_scratch, retrieved, model
    ):
        return True
    return False


def generate_decide_to_talk(
    init_scratch: PersonaScratch,
    init_amem: AssociativeMemory,
    target_scratch: PersonaScratch,
    retrieved: dict[str, list[ConceptNode]],
    model: LLM_API,
):
    x = run_prompt_decide_to_talk(
        init_scratch, init_amem, target_scratch, retrieved, model
    )[0]
    return x == "yes"


def lets_react(init_scratch, target_scratch, retrieved, model):
    if (
        not target_scratch.act_address
        or not target_scratch.act_description
        or not init_scratch.act_address
        or not init_scratch.act_description
    ):
        return False

    if (
        "sleeping" in target_scratch.act_description
        or "sleeping" in init_scratch.act_description
    ):
        return False

    if init_scratch.curr_time.hour == 23:
        return False

    if "waiting" in target_scratch.act_description:
        return False

    # if init_persona.scratch.planned_path == []:
    #   return False

    if init_scratch.act_address != target_scratch.act_address:
        return False

    react_mode = generate_decide_to_react(
        init_scratch, target_scratch, retrieved, model
    )

    if react_mode == "1":
        wait_until = (
            target_scratch.act_start_time
            + datetime.timedelta(minutes=target_scratch.act_duration - 1)
        ).strftime("%B %d, %Y, %H:%M:%S")
        return f"wait: {wait_until}"
    return False


def generate_decide_to_react(
    init_scratch: PersonaScratch,
    target_scratch: PersonaScratch,
    retrieved: EventContext,
    model: LLM_API,
):
    return run_prompt_decide_to_react(init_scratch, target_scratch, retrieved, model)[0]
