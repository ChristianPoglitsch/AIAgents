from typing import Union

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.memory_structures.scratch.user_scratch import UserScratch
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _create_prompt_input(
    iscratch: Union[UserScratch, PersonaScratch],
    tscratch: PersonaScratch,
    statements: str,
):
    prompt_input = [statements, iscratch.name, tscratch.name]
    return prompt_input


def _clean_up_response(response: str):
    return response.split('"')[0].strip()


def _validate_response(output: str):
    try:
        return _clean_up_response(output)
    except BaseException:
        return None


def _get_fail_safe():
    return "..."


def _get_valid_output(model: LLM_API, prompt: AIMessages, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt).strip()
        success = _validate_response(output)
        if success:
            return success
    return _get_fail_safe()


# FIXME: COULD BE BETTER, the prompt is a mess.


def run_prompt_summarize_relationship(
    iscratch: UserScratch,
    tscratch: PersonaScratch,
    model: LLM_API,
    statements: str,
    verbose=False,
):
    prompt_template = (
        BASE_DIR
        + "/LLM_Character/persona/prompt_modules/templates/summarize_chat_relationship.txt"
    )
    prompt_input = _create_prompt_input(iscratch, tscratch, statements)
    prompt = generate_prompt(prompt_input, prompt_template)
    ai_message = AIMessage(message=prompt, role="user", class_type="System", sender=None)
    am = AIMessages()
    am.add_message(ai_message)
    output = _get_valid_output(model, am, COUNTER_LIMIT)

    return output, [output, prompt, prompt_input]


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms
    from LLM_Character.persona.persona import Persona

    person = Persona("FRERO", "nice")

    modelc = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    run_prompt_summarize_relationship(
        person, model, "i will drive to the broeltorens.", "kortrijk"
    )
