from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _create_prompt_input(scratch: PersonaScratch, description: str):
    prompt_input = [scratch.name, scratch.get_str_iss(), scratch.name, description]
    return prompt_input


def _clean_up_response(response: str):
    return int(response.strip())


def _validate_response(output: str):
    try:
        return _clean_up_response(output)
    except BaseException:
        return None


def _get_fail_safe():
    return 4


def _get_valid_output(model, prompt: AIMessages, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt).strip()
        success = _validate_response(output)
        if success:
            return success
    return _get_fail_safe()


def run_prompt_poignancy_chat(
    cscratch: PersonaScratch, description: str, model: LLM_API, verbose=False
):
    prompt_template = (
        BASE_DIR + "/LLM_Character/persona/prompt_modules/templates/poignancy_chat.txt"
    )
    prompt_input = _create_prompt_input(cscratch, description)
    #   example_output = "5" ########
    #   special_instruction = "The output should ONLY contain ONE integer valu
    prompt = generate_prompt(prompt_input, prompt_template)
    am = AIMessages()
    ai_message = AIMessage(message=prompt, role="user", class_type="System", sender=None)  # NOTE: not really user btw
    am.add_message(ai_message)
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
    run_prompt_poignancy_chat(person.scratch, 'desc', model)
