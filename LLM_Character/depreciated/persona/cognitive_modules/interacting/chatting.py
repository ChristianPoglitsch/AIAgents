import datetime
import math

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.persona.cognitive_modules.conversing.ending import (
    _create_react,
    generate_convo_summary,
)
from LLM_Character.persona.cognitive_modules.conversing.reacting import (
    generate_summarize_agent_relationship,
)
from LLM_Character.persona.cognitive_modules.retrieve import retrieve_focal_points
from LLM_Character.persona.memory_structures.associative_memory.associative_memory import (
    AssociativeMemory,
)
from LLM_Character.persona.memory_structures.associative_memory.concept_node import (
    ConceptNode,
)
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.interact_prompts.generate_convo import (
    run_prompt_generative_iterative_personas_chat,
)


def chat_react(
    init_scratch: PersonaScratch,
    init_mem: AssociativeMemory,
    target_scratch: PersonaScratch,
    target_mem: AssociativeMemory,
    model: LLM_API,
):
    # There are two personas -- the persona who is initiating the conversation
    # Actually creating the conversation here.
    convo, duration_min = generate_convo(
        init_scratch, init_mem, target_scratch, target_mem, model
    )
    convo_summary = generate_convo_summary(convo, model)
    inserted_act = convo_summary
    inserted_act_dur = duration_min

    act_start_time = target_scratch.act_start_time

    curr_time = target_scratch.curr_time
    if curr_time.second != 0:
        temp_curr_time = curr_time + datetime.timedelta(seconds=60 - curr_time.second)
        chatting_end_time = temp_curr_time + datetime.timedelta(
            minutes=inserted_act_dur
        )
    else:
        chatting_end_time = curr_time + datetime.timedelta(minutes=inserted_act_dur)

    for i in range(2):
        person, other_person = (
            (init_scratch, target_scratch)
            if i % 2 == 0
            else (target_scratch, init_scratch)
        )
        act_address = f"<persona> {other_person.name}"
        act_event = (person.name, "chat with", other_person.name)
        chatting_with = other_person.name
        chatting_with_buffer = {other_person.name: 800}

        act_obj_description = None
        act_obj_event = (None, None, None)

        _create_react(
            person,
            inserted_act,
            inserted_act_dur,
            act_address,
            act_event,
            chatting_with,
            convo,
            chatting_with_buffer,
            chatting_end_time,
            act_obj_description,
            act_obj_event,
            act_start_time,
        )


def generate_convo(
    init_scratch: PersonaScratch,
    init_mem: AssociativeMemory,
    target_scratch: PersonaScratch,
    target_mem: AssociativeMemory,
    model: LLM_API,
):
    convo = agent_chat_v2(init_scratch, init_mem, target_scratch, target_mem, model)
    all_utt = ""

    for row in convo:
        speaker = row[0]
        utt = row[1]
        all_utt += f"{speaker}: {utt}\n"

    convo_length = math.ceil(int(len(all_utt) / 8) / 30)

    return convo, convo_length


def agent_chat_v2(
    init_scratch: PersonaScratch,
    init_mem: AssociativeMemory,
    target_scratch: PersonaScratch,
    target_mem: AssociativeMemory,
    model: LLM_API,
):
    curr_chat = []

    for i in range(8):
        focal_points = [f"{target_scratch.name}"]
        retrieved = retrieve_focal_points(init_scratch, focal_points, 50)
        relationship = generate_summarize_agent_relationship(
            init_scratch, target_scratch, model, retrieved
        )

        last_chat = ""
        for i in curr_chat[-4:]:
            last_chat += ": ".join(i) + "\n"
        if last_chat:
            focal_points = [
                f"{relationship}",
                f"{target_scratch.name} is {target_scratch.act_description}",
                last_chat,
            ]
        else:
            focal_points = [
                f"{relationship}",
                f"{target_scratch.name} is {target_scratch.act_description}",
            ]
        retrieved = retrieve_focal_points(
            init_scratch, init_mem, focal_points, model, 15
        )
        utt, end = generate_one_utterance(
            init_scratch, init_mem, target_scratch, retrieved, curr_chat, model
        )

        curr_chat += [[init_scratch.name, utt]]
        if end:
            break

        focal_points = [f"{init_scratch.name}"]
        retrieved = retrieve_focal_points(target_scratch, target_mem, focal_points, 50)
        relationship = generate_summarize_agent_relationship(
            target_scratch, init_scratch, retrieved
        )

        last_chat = ""
        for i in curr_chat[-4:]:
            last_chat += ": ".join(i) + "\n"
        if last_chat:
            focal_points = [
                f"{relationship}",
                f"{init_scratch.name} is {init_scratch.act_description}",
                last_chat,
            ]
        else:
            focal_points = [
                f"{relationship}",
                f"{init_scratch.name} is {init_scratch.act_description}",
            ]
        retrieved = retrieve_focal_points(
            target_scratch, target_mem, focal_points, model, 15
        )
        utt, end = generate_one_utterance(
            target_scratch, target_mem, init_scratch, retrieved, curr_chat, model
        )

        curr_chat += [[target_scratch.name, utt]]
        if end:
            break

    return curr_chat


def generate_one_utterance(
    init_scratch: PersonaScratch,
    init_amem: AssociativeMemory,
    target_scratch: PersonaScratch,
    retrieved: dict[str, list[ConceptNode]],
    curr_chat: list[str],
    model: LLM_API,
):
    # Chat version optimized for speed via batch generation
    curr_context = (
        f"{init_scratch.name} "
        + f"was {init_scratch.act_description} "
        + f"when {init_scratch.name} "
        + f"saw {target_scratch.name} "
        + f"in the middle of {target_scratch.act_description}.\n"
    )
    curr_context += (
        f"{init_scratch.name} "
        + "is initiating a conversation with "
        + f"{target_scratch.name}."
    )

    x = run_prompt_generative_iterative_personas_chat(
        init_scratch,
        init_amem,
        target_scratch,
        retrieved,
        curr_chat,
        curr_context,
        model,
    )[0]

    return x["utterance"], x["end"]
