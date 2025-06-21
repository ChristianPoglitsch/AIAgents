from typing import Union

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessages
from LLM_Character.persona.memory_structures.associative_memory.associative_memory import (
    AssociativeMemory,
)
from LLM_Character.persona.memory_structures.associative_memory.concept_node import (
    ConceptNode,
)
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.converse_prompts.generate_iterative_chat import (
    extract_first_json_dict,
)
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _create_prompt_input(
    init_scratch: PersonaScratch,
    init_a_mem: AssociativeMemory,
    target_scratch: PersonaScratch,
    retrieved: dict[str, list[ConceptNode]],
    curr_chat: list[str],
    curr_context: str,
):
    prev_convo_insert = "\n"
    if init_a_mem.seq_chat:
        for i in init_a_mem.seq_chat:
            if i.object == target_scratch.name:
                v1 = int((init_scratch.curr_time - i.created).total_seconds() / 60)
                prev_convo_insert += f"{str(v1)} minutes ago, {init_scratch.name} and {target_scratch.name} were already {i.description} This context takes place after that conversation."
                break
    if prev_convo_insert == "\n":
        prev_convo_insert = ""
    if init_a_mem.seq_chat:
        if (
            int(
                (
                    init_scratch.curr_time - init_a_mem.seq_chat[-1].created
                ).total_seconds()
                / 60
            )
            > 480
        ):
            prev_convo_insert = ""

    curr_sector = f"{init_scratch.get_curr_location()['sector']}"
    curr_arena = f"{init_scratch.get_curr_location()['arena']}"
    curr_location = f"{curr_arena} in {curr_sector}"

    retrieved_str = ""
    for key, vals in retrieved.items():
        for v in vals:
            retrieved_str += f"- {v.description}\n"

    convo_str = ""
    for i in curr_chat:
        convo_str += ": ".join(i) + "\n"
    if convo_str == "":
        convo_str = "[The conversation has not started yet -- start it!]"

    init_iss = f"Here is Here is a brief description of {init_scratch.name}.\n{init_scratch.get_str_iss()}"
    prompt_input = [
        init_iss,
        init_scratch.name,
        retrieved_str,
        prev_convo_insert,
        curr_location,
        curr_context,
        init_scratch.name,
        target_scratch.name,
        convo_str,
        init_scratch.name,
        target_scratch.name,
        init_scratch.name,
        init_scratch.name,
        init_scratch.name,
    ]
    return prompt_input


def _clean_up_response(response: str) -> Union[None, dict[str, str]]:
    obj = extract_first_json_dict(response)
    if not obj:
        return None
    cleaned_dict = dict()
    cleaned = []
    for _, val in obj.items():
        cleaned += [val]

    cleaned_dict["utterance"] = cleaned[0]
    cleaned_dict["end"] = True
    if "f" in str(cleaned[1]).lower():
        cleaned_dict["end"] = False

    return cleaned_dict


def _validate_response(output: str):
    try:
        return _clean_up_response(output)
    except BaseException:
        return None


def _get_fail_safe():
    cleaned_dict = dict()
    cleaned_dict["utterance"] = "..."
    cleaned_dict["end"] = False
    return cleaned_dict


def _get_valid_output(model, prompt, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt)
        if _validate_response(output):
            return _clean_up_response(output)
    return _get_fail_safe()


def run_prompt_generative_iterative_personas_chat(
    init_scratch: PersonaScratch,
    init_amem: AssociativeMemory,
    target_scratch: PersonaScratch,
    retrieved: dict[str, list[ConceptNode]],
    curr_chat: list[str],
    curr_context: str,
    model: LLM_API,
    verbose=False,
):
    prompt_template = (
        BASE_DIR
        + "/LLM_Character/persona/prompt_modules/templates/interative_convo_personas.txt"
    )
    prompt_input = _create_prompt_input(
        init_scratch, init_amem, target_scratch, retrieved, curr_chat, curr_context
    )
    prompt = generate_prompt(prompt_input, prompt_template)

    am = AIMessages()
    am.add_message(prompt, None, "user", "system")

    output = _get_valid_output(model, am, COUNTER_LIMIT)

    return output, [output, prompt, prompt_input]


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms
    from LLM_Character.persona.persona import Persona

    person = Persona("FRERO")

    modelc = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    run_prompt_generative_iterative_personas_chat(
        person, model, "i will drive to the broeltorens.", "kortrijk"
    )
