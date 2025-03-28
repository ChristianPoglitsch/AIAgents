"""
Given the action description, persona,
return the suitable arenas where this action can take place.
"""

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessages
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.memory_structures.spatial_memory import MemoryTree
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _create_prompt_input(
    scratch: PersonaScratch,
    s_mem: MemoryTree,
    act_descrip: str,
    act_world: str,
    act_sector: str,
):
    name = scratch.get_str_name()

    # NOTE world < sectors < arenas < gameobjects
    possible_arenas = s_mem.get_str_accessible_sector_arenas(act_world, act_sector)
    action_description_1, action_description_2 = _decomp_action_desc(act_descrip)

    prompt_input = []
    prompt_input += [name]
    prompt_input += [act_sector]
    prompt_input += [possible_arenas]
    prompt_input += [name]
    prompt_input += [action_description_1]
    prompt_input += [action_description_2]
    prompt_input += [act_sector]
    prompt_input += [possible_arenas]

    return prompt_input


def _decomp_action_desc(action_description: str):
    if "(" in action_description:
        action_description_1 = action_description.split("(")[0].strip()
        action_description_2 = action_description.split("(")[-1][:-1]
        return action_description_1, action_description_2
    else:
        return action_description, action_description


def _clean_up_response(response: str):
    cleaned_response = response.split("}")[0]
    return cleaned_response


def _validate_response(response: str):
    if len(response.strip()) < 1:
        return False
    if "}" not in response:
        return False
    if "," in response:
        return False
    return True


def _get_fail_safe():
    fs = "kitchen"
    return fs


def _get_valid_output(model, prompt, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt)
        if _validate_response(output):
            return _clean_up_response(output)
    return _get_fail_safe()


def run_prompt_action_arena(
    scratch: PersonaScratch,
    s_mem: MemoryTree,
    model: LLM_API,
    action_descrip: str,
    action_world: str,
    action_sector: str,
    verbose=False,
):
    prompt_template = (
        BASE_DIR + "/LLM_Character/persona/prompt_modules/templates/action_arena.txt"
    )
    prompt_input = _create_prompt_input(
        scratch, s_mem, action_descrip, action_world, action_sector
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
    run_prompt_action_arena(
        person, model, "i will drive to the broeltorens.", "kortrijk"
    )
