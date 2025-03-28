"""
Given the persona, returns an integer that indicates the hour when the
persona wakes up.
"""

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessages
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _create_prompt_input(scratch: PersonaScratch) -> list[str]:
    prompt_input = [
        scratch.get_str_iss(),
        scratch.get_str_lifestyle(),
        scratch.get_str_firstname(),
    ]
    return prompt_input


def _clean_up_response(response: str) -> int:
    cr = int(response.strip().lower().split("am")[0])
    return cr


def _validate_response(output: str):
    try:
        return _clean_up_response(output)
    except BaseException:
        return None


def _get_fail_safe() -> int:
    return 8


def _get_valid_output(model: LLM_API, prompt: AIMessages, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt).strip()
        success = _validate_response(output)
        if success:
            return success
    return _get_fail_safe()


def run_prompt_wake_up(scratch: PersonaScratch, model: LLM_API, verbose=False):
    prompt_template = (
        BASE_DIR + "/LLM_Character/persona/prompt_modules/templates/wake_up_hour.txt"
    )
    prompt_input = _create_prompt_input(scratch)
    prompt = generate_prompt(prompt_input, prompt_template)
    am = AIMessages()
    am.add_message(prompt, None, "user", "system")  # NOTE not really user btw
    output = _get_valid_output(model, am, COUNTER_LIMIT)
    # if verbose:
    # print_run_prompts(prompt_template, persona, gpt_param,
    #                   prompt_input, prompt, output)
    return output, [output, prompt, prompt_input]


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms
    from LLM_Character.persona.persona import Persona

    person = Persona("BARA")

    modelc = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    run_prompt_wake_up(person, model)
