import datetime

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessages
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.prompt import generate_prompt
from LLM_Character.util import BASE_DIR

COUNTER_LIMIT = 5


def _get_all_indices(curr_f_org_index: int, amount_activities: int):
    all_indices = []
    for i in range(3):
        if curr_f_org_index + i <= amount_activities:
            all_indices += [curr_f_org_index]
    return all_indices


def _find_start_and_end_time(index: int, f_daily_schedule_hourly):
    start_min = 0
    for i in range(index):
        start_min += f_daily_schedule_hourly[i][1]
    end_min = start_min + f_daily_schedule_hourly[index][1]

    start_time = datetime.datetime.strptime(
        "00:00:00", "%H:%M:%S"
    ) + datetime.timedelta(minutes=start_min)
    end_time = datetime.datetime.strptime("00:00:00", "%H:%M:%S") + datetime.timedelta(
        minutes=end_min
    )

    start_time_str = start_time.strftime("%H:%M%p")
    end_time_str = end_time.strftime("%H:%M%p")

    return start_time_str, end_time_str


def _get_curr_time_and_summary(
    time, all_indices, f_daily_schedule, fullname, curr_f_org_index
):
    curr_time_range = ""
    summ_str = f"Today is {time}. From "
    for index in all_indices:
        start_time_str, end_time_str = _find_start_and_end_time(index, f_daily_schedule)
        summ_str += f"{start_time_str} ~ {end_time_str}, {fullname}\
        is planning on {f_daily_schedule[index][0]}, "
        if curr_f_org_index + 1 == index:
            curr_time_range = f"{start_time_str} ~ {end_time_str}"
    summ_str = summ_str[:-2] + "."
    return summ_str, curr_time_range


def _create_prompt_input(scratch: PersonaScratch, task, duration):
    commonset = scratch.get_str_iss()
    firstname = scratch.get_str_firstname()
    fullname = scratch.get_str_name()
    time = scratch.curr_time.strftime("%B %d, %Y")

    f_daily_schedule = scratch.get_f_daily_schedule_hourly_org()
    amount_activities = len(f_daily_schedule)

    curr_f_org_index = scratch.get_f_daily_schedule_hourly_org_index()
    all_indices = _get_all_indices(curr_f_org_index, amount_activities)
    summ_str, curr_time_range = _get_curr_time_and_summary(
        time, all_indices, f_daily_schedule, fullname, curr_f_org_index
    )
    prompt_input = []
    prompt_input += [commonset]
    prompt_input += [summ_str]
    prompt_input += [firstname]
    prompt_input += [firstname]
    prompt_input += [task]
    prompt_input += [curr_time_range]
    prompt_input += [duration]
    prompt_input += [firstname]
    return prompt_input


# FIXME: check if you can rewrite this peice of junk.


def _clean_up_response(prompt: str, response: str):  # noqa: C901
    temp = [i.strip() for i in response.split("\n")]
    _cr = []
    cr = []
    for count, i in enumerate(temp):
        if count != 0:
            _cr += [" ".join([j.strip() for j in i.split(" ")][3:])]
        else:
            _cr += [i]
    for _, i in enumerate(_cr):
        k = [j.strip() for j in i.split("(duration in minutes:")]
        task = k[0]
        if task[-1] == ".":
            task = task[:-1]
        duration = int(k[1].split(",")[0].strip())
        cr += [[task, duration]]

    total_expected_min = int(
        prompt.split("(total duration in minutes")[-1].split("):")[0].strip()
    )

    # TODO -- now, you need to make sure that this is the same as the sum of
    #         the current action sequence.
    curr_min_slot = [
        ["dummy", -1],
    ]  # (task_name, task_index)
    for count, i in enumerate(cr):
        i_task = i[0]
        i_duration = i[1]

        i_duration -= i_duration % 5
        if i_duration > 0:
            for _ in range(i_duration):
                curr_min_slot += [(i_task, count)]
    curr_min_slot = curr_min_slot[1:]

    if len(curr_min_slot) > total_expected_min:
        last_task = curr_min_slot[60]
        for i in range(1, 6):
            curr_min_slot[-1 * i] = last_task
    elif len(curr_min_slot) < total_expected_min:
        last_task = curr_min_slot[-1]
        for _ in range(total_expected_min - len(curr_min_slot)):
            curr_min_slot += [last_task]

    cr_ret = [
        ["dummy", -1],
    ]
    for task, _ in curr_min_slot:
        if task != cr_ret[-1][0]:
            cr_ret += [[task, 1]]
        else:
            cr_ret[-1][1] += 1
    cr = cr_ret[1:]

    return cr


def _validate_response(prompt: str, response: str):
    try:
        _clean_up_response(prompt, response)
    except BaseException:
        return False
    return True


def _get_fail_safe():
    # FIXME this needs to be changed,
    # repo where this is forked from gave wrong faile safe...
    fs = [("asleep", 1)]
    return fs


def _get_valid_output(model, prompt, counter_limit):
    for _ in range(counter_limit):
        output = model.query_text(prompt)
        if _validate_response(prompt, output):
            return _clean_up_response(prompt, output)
    return _get_fail_safe()


def run_prompt_task_decomp(
    scratch: PersonaScratch, model: LLM_API, task, duration, verbose=False
):
    prompt_template = (
        BASE_DIR + "/LLM_Character/persona/prompt_modules/templates/task_decomp.txt"
    )
    prompt_input = _create_prompt_input(scratch, task, duration)
    prompt = generate_prompt(prompt_input, prompt_template)

    am = AIMessages()
    am.add_message(prompt, None, "user", "system")

    output = _get_valid_output(model, am, COUNTER_LIMIT)

    # FIXME: TRY TO REWRITE THIS JUNK AS WELL.
    fin_output = []
    time_sum = 0
    for i_task, i_duration in output:
        time_sum += i_duration
        # HM?????????
        # if time_sum < duration:
        if time_sum <= duration:
            fin_output += [[i_task, i_duration]]
        else:
            break
    ftime_sum = 0
    for _, fi_duration in fin_output:
        ftime_sum += fi_duration

    # print ("for debugging... line 365", fin_output)
    fin_output[-1][1] += duration - ftime_sum
    output = fin_output

    task_decomp = output
    ret = []
    for decomp_task, duration in task_decomp:
        ret += [[f"{task} ({decomp_task})", duration]]
    output = ret

    return output, [output, prompt, prompt_input]


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms
    from LLM_Character.persona.persona import Persona

    person = Persona("FRERO")

    modelc = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    run_prompt_task_decomp(person, model, "i will drive to the broeltorens.", 5)
