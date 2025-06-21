"""
The long term planning that spans a day.
"""

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessages
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _create_prompt_input(scratch: PersonaScratch, wake_up_hour: int) -> list[str]:
    prompt_input = []
    prompt_input += [scratch.get_str_iss()]
    prompt_input += [scratch.get_str_lifestyle()]
    prompt_input += [scratch.get_str_curr_date_str()]
    prompt_input += [scratch.get_str_firstname()]
    prompt_input += [f"{str(wake_up_hour)}:00 am"]
    return prompt_input


def _clean_up_response(response):
    cr = []
    _cr = response.split(")")
    for i in _cr:
        if i[-1].isdigit():
            i = i[:-1].strip()
            if i[-1] == "." or i[-1] == ",":
                cr += [i[:-1].strip()]
    return cr


def _validate_response(response: str) -> bool:
    try:
        _clean_up_response(response)
    except BaseException:
        return False
    return True


def _get_fail_safe():
    fs = [
        "wake up and complete the morning routine at 6:00 am",
        "eat breakfast at 7:00 am",
        "read a book from 8:00 am to 12:00 pm",
        "have lunch at 12:00 pm",
        "take a nap from 1:00 pm to 4:00 pm",
        "relax and watch TV from 7:00 pm to 8:00 pm",
        "go to bed at 11:00 pm",
    ]
    return fs


def _get_valid_output(model, prompt, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt)
        if _validate_response(output):
            return _clean_up_response(output)
    return _get_fail_safe()


def run_prompt_daily_plan(
    scratch: PersonaScratch, wake_up_hour: int, model: LLM_API, verbose=False
):
    prompt_template = (
        BASE_DIR + "/LLM_Character/persona/prompt_modules/templates/daily_plan.txt"
    )
    prompt_input = _create_prompt_input(scratch, wake_up_hour)
    prompt = generate_prompt(prompt_input, prompt_template)

    am = AIMessages()
    am.add_message(prompt, None, "user", "system")  # NOTE: not really user btw

    output = _get_valid_output(model, am, COUNTER_LIMIT)
    output = [
        f"wake up and complete the morning routine at {wake_up_hour}:00 am"
    ] + output
    return output, [output, prompt, prompt_input]


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms
    from LLM_Character.persona.persona import Persona

    person = Persona("BARA")

    modelc = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    run_prompt_daily_plan(person, 8, model)
