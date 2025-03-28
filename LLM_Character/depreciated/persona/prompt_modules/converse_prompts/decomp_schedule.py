import datetime

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _create_prompt_input(
    cscratch: PersonaScratch,
    main_act_dur,
    truncated_act_dur,
    start_time_hour,
    end_time_hour,
    inserted_act,
    inserted_act_dur,
):
    persona_name = cscratch.name
    start_hour_str = start_time_hour.strftime("%H:%M %p")
    end_hour_str = end_time_hour.strftime("%H:%M %p")

    original_plan = ""
    for_time = start_time_hour
    for i in main_act_dur:
        original_plan += (
            f'{for_time.strftime("%H:%M")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- '
            + i[0]
        )
        original_plan += "\n"
        for_time += datetime.timedelta(minutes=int(i[1]))

    new_plan_init = ""
    for_time = start_time_hour
    for count, i in enumerate(truncated_act_dur):
        new_plan_init += (
            f'{for_time.strftime("%H:%M")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- '
            + i[0]
        )
        new_plan_init += "\n"
        if count < len(truncated_act_dur) - 1:
            for_time += datetime.timedelta(minutes=int(i[1]))

    # FIXME: variable i possibly unbound was ook zo in de originele repo...
    # dus check wat die waarde zal zijn wss None...
    new_plan_init += (for_time + datetime.timedelta(minutes=int(i[1]))).strftime(
        "%H:%M"
    ) + " ~"

    prompt_input = [
        persona_name,
        start_hour_str,
        end_hour_str,
        original_plan,
        persona_name,
        inserted_act,
        inserted_act_dur,
        persona_name,
        start_hour_str,
        end_hour_str,
        end_hour_str,
        new_plan_init,
    ]
    return prompt_input


def _clean_up_response(prompt: str, response: str):
    new_schedule = prompt + " " + response.strip()
    new_schedule = new_schedule.split("The revised schedule:")[-1].strip()
    new_schedule = new_schedule.split("\n")

    ret_temp = []
    for i in new_schedule:
        ret_temp += [i.split(" -- ")]

    ret = []
    for time_str, action in ret_temp:
        start_time = time_str.split(" ~ ")[0].strip()
        end_time = time_str.split(" ~ ")[1].strip()
        delta = datetime.datetime.strptime(
            end_time, "%H:%M"
        ) - datetime.datetime.strptime(start_time, "%H:%M")
        delta_min = int(delta.total_seconds() / 60)
        if delta_min < 0:
            delta_min = 0
        ret += [[action, delta_min]]
    return ret


def _validate_response(prompt: str, output: str):
    try:
        gpt_response = _clean_up_response(prompt, output)
        dur_sum = 0
        for act, dur in gpt_response:
            dur_sum += dur
            if str(type(act)) != "<class 'str'>":
                return False
            if str(type(dur)) != "<class 'int'>":
                return False
        x = (
            prompt.split("\n")[0]
            .split("originally planned schedule from")[-1]
            .strip()[:-1]
        )
        x = [datetime.datetime.strptime(i.strip(), "%H:%M %p") for i in x.split(" to ")]
        delta_min = int((x[1] - x[0]).total_seconds() / 60)

        if int(dur_sum) != int(delta_min):
            return False

    except BaseException:
        return False
    return True


def _get_fail_safe(main_act_dur, truncated_act_dur):
    dur_sum = 0
    for act, dur in main_act_dur:
        dur_sum += dur

    ret = truncated_act_dur[:]
    ret += main_act_dur[len(ret) - 1 :]

    ret_dur_sum = 0
    count = 0
    over = None
    for act, dur in ret:
        ret_dur_sum += dur
        if ret_dur_sum == dur_sum:
            break
        if ret_dur_sum > dur_sum:
            over = ret_dur_sum - dur_sum
            break
        count += 1

    if over:
        ret = ret[: count + 1]
        ret[-1][1] -= over

    return ret


def _get_valid_output(model, prompt, main_act_dur, truncated_act_dur, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt).strip()
        success = _validate_response(prompt, output)
        if success:
            return success
    return _get_fail_safe(main_act_dur, truncated_act_dur)


def run_prompt_decomp_schedule(
    cscratch: PersonaScratch,
    main_act_dur,
    truncated_act_dur,
    start_time_hour,
    end_time_hour,
    inserted_act,
    inserted_act_dur,
    model: LLM_API,
    verbose=False,
):
    prompt_template = (
        BASE_DIR + "/LLM_Character/persona/prompt_modules/templates/decomp_schedule.txt"
    )
    prompt_input = _create_prompt_input(
        cscratch,
        main_act_dur,
        truncated_act_dur,
        start_time_hour,
        end_time_hour,
        inserted_act,
        inserted_act_dur,
    )
    prompt = generate_prompt(prompt_input, prompt_template)
    output = _get_valid_output(
        model, prompt, main_act_dur, truncated_act_dur, COUNTER_LIMIT
    )

    return output, [output, prompt, prompt_input]


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms
    from LLM_Character.persona.persona import Persona

    person = Persona("FRERO", "nice")

    modelc = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    # run_prompt_decomp_schedule(person, model, "i will drive to the broeltorens.", "kortrijk" )
