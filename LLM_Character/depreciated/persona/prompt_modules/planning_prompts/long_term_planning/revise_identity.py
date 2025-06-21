"""
The long term planning that spans a day.
"""

import datetime

import LLM_Character.persona.memory_structures.scratch.persona_scratch as PersonaScratch
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessages
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _component_statements(retrieved):
    statements = ""
    for _, val in retrieved.items():
        for i in val:
            statements += (
                f"{i.created.strftime('%A %B %d -- %H:%M %p')}: {i.embedding_key}\n"
            )
    return statements


def _create_prompt_input_1(scratch: PersonaScratch, retrieved) -> list[str]:
    stmts = _component_statements(retrieved)
    p_name = scratch.name
    time = scratch.curr_time.strftime("%A %B %d")

    plan_input = []
    plan_input += [stmts]
    plan_input += [p_name]
    plan_input += [time]
    return plan_input


def _create_prompt_input_2(scratch: PersonaScratch, retrieved) -> list[str]:
    stmts = _component_statements(retrieved)
    p_name = scratch.name

    thought_prompt = []
    thought_prompt += [stmts]
    thought_prompt += [p_name]
    return thought_prompt


def _create_prompt_input_3(
    scratch: PersonaScratch, plan_note: str, thought_note: str, retrieved
) -> list[str]:
    stmts = _component_statements(retrieved)
    p_name = scratch.name
    time = scratch.curr_time.strftime("%A %B %d")
    time_diff = (scratch.curr_time - datetime.timedelta(days=1)).strftime("%A %B %d")
    currently = scratch.currently
    notes = (plan_note + thought_note).replace("\n", "")

    currently_prompt = []
    currently_prompt += [stmts]
    currently_prompt += [p_name]
    currently_prompt += [time_diff]
    currently_prompt += [currently]
    currently_prompt += [notes]
    currently_prompt += [time]

    return currently_prompt


def _create_prompt_input_4(scratch: PersonaScratch) -> list[str]:
    commonset = scratch.get_str_iss()
    curr_time = scratch.curr_time.strftime("%A %B %d")
    p_name = scratch.name

    daily_req_prompt = []
    daily_req_prompt += [commonset]
    daily_req_prompt += [curr_time]
    daily_req_prompt += [p_name]

    return daily_req_prompt


def _get_valid_output(model, prompt, counter_limit):
    return model.query_text(prompt)


def run_prompt_revise_identity(
    scratch: PersonaScratch, model: LLM_API, retrieved, verbose=False
):
    prompt_template_1 = (
        BASE_DIR
        + "/LLM_Character/persona/prompt_modules/templates/revise_identity_1.txt"
    )
    prompt_input1 = _create_prompt_input_1(scratch, retrieved)
    prompt1 = generate_prompt(prompt_input1, prompt_template_1)
    am = AIMessages()
    am.add_message(prompt1, None, "user", "system")  # NOTE not really user btw
    plan_note = _get_valid_output(model, am, COUNTER_LIMIT)

    prompt_template_2 = (
        BASE_DIR
        + "/LLM_Character/persona/prompt_modules/templates/revise_identity_2.txt"
    )
    prompt_input2 = _create_prompt_input_2(scratch, retrieved)
    prompt2 = generate_prompt(prompt_input2, prompt_template_2)
    am = AIMessages()
    am.add_message(prompt2, None, "user", "system")
    thought_note = _get_valid_output(model, am, COUNTER_LIMIT)

    prompt_template_3 = (
        BASE_DIR
        + "/LLM_Character/persona/prompt_modules/templates/revise_identity_3.txt"
    )
    prompt_input3 = _create_prompt_input_3(scratch, plan_note, thought_note, retrieved)
    prompt3 = generate_prompt(prompt_input3, prompt_template_3)
    am = AIMessages()
    am.add_message(prompt3, None, "user", "system")
    currently_note = _get_valid_output(model, am, COUNTER_LIMIT)

    prompt_template_4 = (
        BASE_DIR
        + "/LLM_Character/persona/prompt_modules/templates/revise_identity_4.txt"
    )
    prompt_input4 = _create_prompt_input_4(scratch)
    prompt4 = generate_prompt(prompt_input4, prompt_template_4)
    am = AIMessages()
    am.add_message(prompt4, None, "user", "system")
    daily_req_note = _get_valid_output(model, am, COUNTER_LIMIT)

    new_daily_req = daily_req_note.replace("\n", " ")
    return plan_note, thought_note, currently_note, new_daily_req


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms
    from LLM_Character.persona.persona import Persona

    person = Persona("BARA")

    modelc = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)

    # TODO replace retrived with actual dictionary with some list of
    # conceptNodes as values
    retrieved = None
    run_prompt_revise_identity(person, model, retrieved)
