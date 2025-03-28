"""
Given the action description, persona,
return the suitable tuple (subject, predicate, object) of the action description.
"""

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _create_prompt_input(scratch: PersonaScratch, action_description: str):
    p_name = scratch.get_str_name()

    if "(" in action_description:
        action_description = action_description.split("(")[-1].split(")")[0]

    prompt_input = [p_name, action_description, p_name]
    return prompt_input


def _clean_up_response(response):
    cr = response.strip()
    cr = [i.strip() for i in cr.split(")")[0].split(",")]
    return cr


def _validate_response(response):
    try:
        response = _clean_up_response(response)
        if len(response) != 2:
            return False
    except BaseException:
        return False
    return True


def _get_fail_safe(scratch: PersonaScratch):
    p_name = scratch.get_str_name()
    fs = (p_name, "is", "idle")
    return fs


def _get_valid_output(scratch: PersonaScratch, model, prompt, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt)
        if _validate_response(output):
            return _clean_up_response(output)
    return _get_fail_safe(scratch)


def run_prompt_event_triple(
    scratch: PersonaScratch, model: LLM_API, action_description: str, verbose=False
):
    prompt_template = (
        BASE_DIR
        + "/LLM_Character/persona/prompt_modules/templates/generate_event_triple.txt"
    )
    prompt_input = _create_prompt_input(scratch, action_description)
    prompt = generate_prompt(prompt_input, prompt_template)

    am = AIMessages()
    am.add_message(AIMessage(prompt, None, "user", "system"))

    output = _get_valid_output(scratch, model, am, COUNTER_LIMIT)

    p_name = scratch.get_str_name()
    output = (p_name, output[0], output[1])

    return output, [output, prompt, prompt_input]


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms
    from LLM_Character.persona.persona import Persona

    person = Persona("FRERO")

    modelc = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    run_prompt_event_triple(person, model, "i will drive to the broeltorens.")
