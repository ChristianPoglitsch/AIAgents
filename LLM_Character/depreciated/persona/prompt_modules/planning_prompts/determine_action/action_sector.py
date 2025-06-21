"""
Given the action description, persona,
return the suitable sector where this action can take place.
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
    scratch: PersonaScratch, s_mem: MemoryTree, action_description: str
):
    act_world = scratch.get_curr_location()["world"]
    act_sector = scratch.get_curr_location()["sector"]
    liv_sector = scratch.get_living_area()["sector"]
    name = scratch.get_str_name()

    # NOTE world < sectors < arenas < gameobjects
    possible_arenas1 = s_mem.get_str_accessible_sector_arenas(act_world, liv_sector)
    possible_arenas2 = s_mem.get_str_accessible_sector_arenas(act_world, act_sector)
    possible_sectors = s_mem.get_str_accessible_sectors(act_world)

    daily_plan = scratch.get_str_daily_plan_req()
    action_description_1, action_description_2 = _decomp_action_desc(action_description)

    prompt_input = []
    prompt_input += [name]
    prompt_input += [liv_sector]
    prompt_input += [possible_arenas1]
    prompt_input += [name]
    prompt_input += [act_sector]
    prompt_input += [possible_arenas2]
    prompt_input += [f"\n{daily_plan}"]
    prompt_input += [possible_sectors]

    prompt_input += [name]
    prompt_input += [action_description_1]

    prompt_input += [action_description_2]
    prompt_input += [name]

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


def run_prompt_action_sector(
    scratch: PersonaScratch,
    s_mem: MemoryTree,
    model: LLM_API,
    action_description: str,
    verbose=False,
):
    prompt_template = (
        BASE_DIR + "/LLM_Character/persona/prompt_modules/templates/action_sector.txt"
    )
    prompt_input = _create_prompt_input(scratch, s_mem, action_description)
    prompt = generate_prompt(prompt_input, prompt_template)

    am = AIMessages()
    am.add_message(prompt, None, "user", "system")

    output = _get_valid_output(model, am, COUNTER_LIMIT)

    y = scratch.get_curr_location()["world"]
    x = [i.strip() for i in s_mem.get_str_accessible_sectors(y).split(",")]
    if output not in x:
        output = scratch.get_living_area()["sector"]

    return output, [output, prompt, prompt_input]


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms
    from LLM_Character.persona.persona import Persona

    person = Persona("FRERO")

    modelc = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    run_prompt_action_sector(person, model, "i will drive to the broeltorens.")
