from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _create_prompt_input(n: int, statements: str):
    prompt_input = [statements, str(n)]
    return prompt_input


def _clean_up_response(response: str):
    response = "1) " + response.strip()
    ret = []
    for i in response.split("\n"):
        ret += [i.split(") ")[-1]]
    return ret


def _validate_response(response: str):
    try:
        return _clean_up_response(response)
    except BaseException:
        return None


def _get_fail_safe(n):
    return ["Who am I"] * n


# FIXME change in each for loop the paramters of the model for example, like in the other repo.
# if needed


def _get_valid_output(model, prompt, n, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt)
        if _validate_response(output):
            return _clean_up_response(output)
    return _get_fail_safe(n)


def run_prompt_generate_focal_pt(model: LLM_API, n: int, all_utt: str, verbose=False):
    prompt_template = (
        BASE_DIR
        + "/LLM_Character/persona/prompt_modules/templates/generate_focal_pt.txt"
    )
    prompt_input = _create_prompt_input(n, all_utt)
    prompt = generate_prompt(prompt_input, prompt_template)
    # example_output = '["What should Jane do for lunch", "Does Jane like stra
    # special_instruction = "Output must be a list of str." ########
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
    run_prompt_generate_focal_pt(
        person.scratch, model, "i will drive to the broeltorens."
    )
