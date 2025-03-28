import re

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _create_prompt_input(n: int, statements: str):
    prompt_input = [statements, str(n)]
    return prompt_input


def _clean_up_response(response: str):
    response = "1. " + response.strip()
    ret = dict()
    for i in response.split("\n"):
        row = i.split(". ")[-1]
        thought = row.split("(because of ")[0].strip()
        evi_raw = row.split("(because of ")[1].split(")")[0].strip()
        evi_raw = re.findall(r"\d+", evi_raw)
        evi_raw = [int(i.strip()) for i in evi_raw]
        ret[thought] = evi_raw
    return ret


def _validate_response(response: str):
    try:
        return _clean_up_response(response)
    except BaseException:
        return None


def _get_fail_safe(n: int):
    return ["I am hungry"] * n


def _get_valid_output(model, prompt, n, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt)
        if _validate_response(output):
            return _clean_up_response(output)
    return _get_fail_safe(n)


def run_prompt_insight_and_evidence(
    model: LLM_API, n: int, all_utt: str, verbose=False
):
    prompt_template = (
        BASE_DIR
        + "/LLM_Character/persona/prompt_modules/templates/insight_and_evidence.txt"
    )
    prompt_input = _create_prompt_input(n, all_utt)
    prompt = generate_prompt(prompt_input, prompt_template)
    am = AIMessages()
    am.add_message(AIMessage(prompt, None, "user", "system"))  # NOTE: not really user btw
    output = _get_valid_output(model, am, n, COUNTER_LIMIT)

    return output, [output, prompt, prompt_input]


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms
    from LLM_Character.persona.persona import Persona

    person = Persona("FRERO", "nice")

    modelc = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    run_prompt_insight_and_evidence(model, 3, "i will drive to the broeltorens.")
