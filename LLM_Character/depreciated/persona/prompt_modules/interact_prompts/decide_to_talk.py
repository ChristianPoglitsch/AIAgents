from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessages
from LLM_Character.persona.cognitive_modules.retrieve import EventContext
from LLM_Character.persona.memory_structures.associative_memory.associative_memory import (
    AssociativeMemory,
)
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _create_prompt_input(
    init_scratch: PersonaScratch,
    init_amem: AssociativeMemory,
    target_scratch: PersonaScratch,
    retrieved: EventContext,
):
    last_chat = init_amem.get_last_chat(target_scratch.name)
    last_chatted_time = ""
    last_chat_about = ""
    if last_chat:
        last_chatted_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
        last_chat_about = last_chat.description

    context = ""
    for c_node in retrieved["events"]:
        curr_desc = c_node.description.split(" ")
        curr_desc[2:3] = ["was"]
        curr_desc = " ".join(curr_desc)
        context += f"{curr_desc}. "
    context += "\n"
    for c_node in retrieved["thoughts"]:
        context += f"{c_node.description}. "

    curr_time = init_scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    init_act_desc = init_scratch.act_description
    if "(" in init_act_desc:
        init_act_desc = init_act_desc.split("(")[-1][:-1]

    if len(init_scratch.planned_path) == 0 and "waiting" not in init_act_desc:
        init_p_desc = f"{init_scratch.name} is already {init_act_desc}"
    elif "waiting" in init_act_desc:
        init_p_desc = f"{init_scratch.name} is {init_act_desc}"
    else:
        init_p_desc = f"{init_scratch.name} is on the way to {init_act_desc}"

    target_act_desc = target_scratch.act_description
    if "(" in target_act_desc:
        target_act_desc = target_act_desc.split("(")[-1][:-1]

    if len(target_scratch.planned_path) == 0 and "waiting" not in init_act_desc:
        target_p_desc = f"{target_scratch.name} is already {target_act_desc}"
    elif "waiting" in init_act_desc:
        target_p_desc = f"{init_scratch.name} is {init_act_desc}"
    else:
        target_p_desc = f"{target_scratch.name} is on the way to {target_act_desc}"

    prompt_input = []
    prompt_input += [context]

    prompt_input += [curr_time]

    prompt_input += [init_scratch.name]
    prompt_input += [target_scratch.name]
    prompt_input += [last_chatted_time]
    prompt_input += [last_chat_about]

    prompt_input += [init_p_desc]
    prompt_input += [target_p_desc]
    prompt_input += [init_scratch.name]
    prompt_input += [target_scratch.name]
    return prompt_input


def _clean_up_response(response: str):
    return response.split("Answer in yes or no:")[-1].strip().lower()


def _validate_response(response: str):
    try:
        return _clean_up_response(response) in ["yes", "no"]
    except BaseException:
        return False


def _get_fail_safe():
    return "yes"


def _get_valid_output(model, prompt, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt).strip()
        success = _validate_response(output)
        if success:
            return success
    return _get_fail_safe()


def run_prompt_decide_to_talk(
    init_scratch: PersonaScratch,
    init_amem: AssociativeMemory,
    target_scratch: PersonaScratch,
    retrieved: EventContext,
    model: LLM_API,
    verbose=False,
):
    prompt_template = (
        BASE_DIR + "/LLM_Character/persona/prompt_modules/templates/decide_to_talk.txt"
    )
    prompt_input = _create_prompt_input(
        init_scratch, init_amem, target_scratch, retrieved
    )
    prompt = generate_prompt(prompt_input, prompt_template)

    am = AIMessages()
    am.add_message(prompt, None, "user", "system")

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
    # run_prompt_decide_to_talk(person, model, "i will drive to the broeltorens.", "kortrijk" )
