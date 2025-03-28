import json
from typing import Union

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.persona.memory_structures.associative_memory.associative_memory import (  # noqa: E501
    AssociativeMemory,
)
from LLM_Character.persona.memory_structures.associative_memory.concept_node import (
    ConceptNode,
)
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.memory_structures.scratch.user_scratch import UserScratch
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _create_prompt_input_1(  # noqa: C901
    uscratch: UserScratch,
    cscratch: PersonaScratch,
    ca_mem: AssociativeMemory,
    retrieved: dict[str, list[ConceptNode]],
    curr_context: str,
    curr_chat: list[AIMessage],
    ending: str,
):
    prev_convo_insert = "\n"
    if ca_mem.seq_chat:
        for i in ca_mem.seq_chat:
            if i.object == uscratch.name:
                v1 = int((cscratch.curr_time - i.created).total_seconds() / 60)
                prev_convo_insert += f"{str(v1)} minutes ago, {cscratch.name} and \
                {uscratch.name} were already {i.description} \
                This context takes place after that conversation."
                for _, vals in retrieved.items():
                    for v in vals:
                        for row in v.filling:
                            prev_convo_insert += f"{row[0]}: {row[1]}\n"
                break
    if prev_convo_insert == "\n":
        prev_convo_insert = "You don't know each other"
    if ca_mem.seq_chat:
        if (
            int((cscratch.curr_time - ca_mem.seq_chat[-1].created).total_seconds() / 60)
            > 480
        ):
            prev_convo_insert = "You know each other"

    curr_location = None
    if(cscratch.get_curr_location() != None):
        curr_sector = f"{cscratch.get_curr_location()['sector']}"
        curr_arena = f"{cscratch.get_curr_location()['arena']}"
        curr_location = f"{curr_arena} in {curr_sector}"

    retrieved_str = ""
    for _, vals in retrieved.items():
        for v in vals:
            retrieved_str += f"- {v.description}\n"

    convo_str = ""
    for i in curr_chat:
        convo_str += i.print_message_sender() + "\n"

    if convo_str == "":
        convo_str = "[The conversation has not started yet -- start it!]"

    if uscratch.name in cscratch.curr_trust.keys():
        trust_str = str(cscratch.curr_trust[uscratch.name])
    else:
        trust_str = "5"

    init_iss = (
        f"Here is a brief description of {cscratch.name}.\n{cscratch.get_str_iss()}"
    )
    prompt_input = [
        init_iss,
        cscratch.name,
        retrieved_str,
        prev_convo_insert,
        curr_location,
        curr_context,
        uscratch.name,
        convo_str,
        cscratch.curr_emotion,
        trust_str,
        ending,
    ]
    return prompt_input


def _create_prompt_input_2(cscratch: PersonaScratch, curr_chat: list[AIMessage]):
    convo_str = ""
    for i in curr_chat:
        convo_str += i.print_message_sender() + "\n"

    if convo_str == "":
        convo_str = "[The conversation has not started yet -- start it!]"

    prompt_input = [convo_str, cscratch.name]
    return prompt_input


def _clean_up_response_1(response: str) -> Union[None, dict[str, str]]:
    obj = extract_first_json_dict(response)
    if not obj:
        return None
    cleaned_dict = {}
    cleaned = []
    for _, val in obj.items():
        cleaned += [val]

    cleaned_dict["utterance"] = cleaned[0]
    cleaned_dict["trust"] = int(cleaned[1])
    cleaned_dict["emotion"] = cleaned[2]
    return cleaned_dict


def _clean_up_response_2(response: str) -> Union[None, dict[str, str]]:
    obj = extract_first_json_dict(response)
    if not obj:
        return None
    cleaned_dict = {}
    cleaned = []
    for _, val in obj.items():
        cleaned += [val]
    cleaned_dict["end"] = True
    if "f" in str(cleaned[0]) or "F" in str(cleaned[0]):
        cleaned_dict["end"] = False
    return cleaned_dict


def extract_first_json_dict(data_str: str) -> Union[None, dict[str, str]]:
    start_idx = data_str.find("{")
    end_idx = data_str.find("}", start_idx) + 1

    if start_idx == -1 or end_idx == 0:
        return None

    json_str = data_str[start_idx:end_idx]
    try:
        json_dict = json.loads(json_str)
        return json_dict
    except json.JSONDecodeError:
        return None


def _validate_response(output: str, clean_up_response: callable):
    try:
        return clean_up_response(output)
    except BaseException:
        return None


def _get_fail_safe():
    return {"utterance": "...", "end": False}


def _get_valid_output(
    model: LLM_API, prompt, clean_up_response: callable, counter_limit
):
    for _ in range(counter_limit):
        output = model.query_text(prompt).strip()
        success = _validate_response(output, clean_up_response)
        if success:
            return success
    return _get_fail_safe()


def run_prompt_iterative_chat(
    uscratch: UserScratch,
    cscratch: PersonaScratch,
    camem: AssociativeMemory,
    model: LLM_API,
    retrieved: dict[str, list[ConceptNode]],
    curr_context: str,
    curr_chat: list[AIMessage],
    verbose=False,
) -> Union[str, dict[str, str]]:
    prompt_template = (
        BASE_DIR + "/LLM_Character/persona/prompt_modules/templates/is_convo_ending.txt"
    )
    prompt_input = _create_prompt_input_2(cscratch, curr_chat)
    prompt = generate_prompt(prompt_input, prompt_template)

    am = AIMessages()
    ai_message = AIMessage(message=prompt, role="user", class_type="System", sender=None)
    am.add_message(ai_message)
    output2 = _get_valid_output(model, am, _clean_up_response_2, COUNTER_LIMIT)
    prompt_template = (
        BASE_DIR + "/LLM_Character/persona/prompt_modules/templates/iterative_convo.txt"
    )
    prompt_input = _create_prompt_input_1(
        uscratch, cscratch, camem, retrieved, curr_context, curr_chat, output2["end"]
    )
    prompt = generate_prompt(prompt_input, prompt_template)

    am = AIMessages()
    ai_message = AIMessage(message=prompt, role="user", class_type="System", sender=None)
    am.add_message(ai_message)
    output1 = _get_valid_output(model, am, _clean_up_response_1, COUNTER_LIMIT)

    return output1, output2
