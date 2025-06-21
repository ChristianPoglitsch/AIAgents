from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessages
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR, get_random_alphanumeric

COUNTER_LIMIT = 5


def _create_prompt_input(
    scratch: PersonaScratch,
    curr_hour_str: str,
    p_f_ds_hourly_org: list[str],
    hour_str: list[str],
    intermission2=None,
) -> list[str]:
    schedule_format = ""
    for i in hour_str:
        schedule_format += f"[{scratch.get_str_curr_date_str()} -- {i}]"
        schedule_format += " Activity: [Fill in]\n"
    schedule_format = schedule_format[:-1]

    intermission_str = "Here the originally intended hourly breakdown of"
    intermission_str += f" {scratch.get_str_firstname()}'s schedule today: "
    for count, i in enumerate(scratch.daily_req):
        intermission_str += f"{str(count+1)}) {i}, "
    intermission_str = intermission_str[:-2]

    prior_schedule = ""
    if p_f_ds_hourly_org:
        prior_schedule = "\n"
        for count, i in enumerate(p_f_ds_hourly_org):
            prior_schedule += f"[(ID:{get_random_alphanumeric()})"
            prior_schedule += f" {scratch.get_str_curr_date_str()} --"
            prior_schedule += f" {hour_str[count]}] Activity:"
            prior_schedule += f" {scratch.get_str_firstname()}"
            prior_schedule += f" is {i}\n"

    prompt_ending = f"[(ID:{get_random_alphanumeric()})"
    prompt_ending += f" {scratch.get_str_curr_date_str()}"
    prompt_ending += f" -- {curr_hour_str}] Activity:"
    prompt_ending += f" {scratch.get_str_firstname()} is"

    if intermission2:
        intermission2 = f"\n{intermission2}"

    prompt_input = []
    prompt_input += [schedule_format]
    prompt_input += [scratch.get_str_iss()]

    prompt_input += [prior_schedule + "\n"]
    prompt_input += [intermission_str]
    if intermission2:
        prompt_input += [intermission2]
    else:
        prompt_input += [""]
    prompt_input += [prompt_ending]

    return prompt_input


def _clean_up_response(response):
    cr = response.strip()
    if cr[-1] == ".":
        cr = cr[:-1]
    return cr


def _validate_response(output: str):
    try:
        return _clean_up_response(output)
    except BaseException:
        return None


def _get_fail_safe():
    return "asleep"


def _get_valid_output(model, prompt, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt)
        if _validate_response(output):
            return _clean_up_response(output)
    return _get_fail_safe()


def run_prompt_hourly_schedule(
    scratch: PersonaScratch,
    curr_hour_str: str,
    p_f_ds_hourly_org: list[str],
    hour_str: list[str],
    model: LLM_API,
    intermission2=None,
    verbose=False,
):
    prompt_template = (
        BASE_DIR
        + "/LLM_Character/persona/prompt_modules/templates/generate_hourly_schedule.txt"
    )
    prompt_input = _create_prompt_input(
        scratch, curr_hour_str, p_f_ds_hourly_org, hour_str, intermission2
    )
    prompt = generate_prompt(prompt_input, prompt_template)
    am = AIMessages()
    am.add_message(prompt, None, "user", "system")
    output = _get_valid_output(model, am, COUNTER_LIMIT)
    return output, [output, prompt, prompt_input]


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms
    from LLM_Character.persona.persona import Persona

    person = Persona("BARA")

    modelc = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    # run_prompt_hourly_schedule(person, 8, model)
