import datetime
import logging
from typing import Union

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.persona.cognitive_modules.retrieve import retrieve_focal_points
from LLM_Character.persona.memory_structures.associative_memory.associative_memory import (  # noqa: E501
    AssociativeMemory,
)
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.planning_prompts.long_term_planning.daily_plan import (  # noqa: E501
    run_prompt_daily_plan,
)
from LLM_Character.persona.prompt_modules.planning_prompts.long_term_planning.hourly_schedule import (  # noqa: E501
    run_prompt_hourly_schedule,
)
from LLM_Character.persona.prompt_modules.planning_prompts.long_term_planning.revise_identity import (  # noqa: E501
    run_prompt_revise_identity,
)
from LLM_Character.persona.prompt_modules.planning_prompts.long_term_planning.wake_up import (  # noqa: E501
    run_prompt_wake_up,
)
from LLM_Character.util import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def _long_term_planning(
    scratch: PersonaScratch,
    a_mem: AssociativeMemory,
    new_day: Union[str, None],
    model: LLM_API,
):
    wake_up_hour = generate_wake_up_hour(scratch, model)
    if new_day == "First day":
        logger.info("Generating first daily plan")
        scratch.daily_req = generate_first_daily_plan(scratch, wake_up_hour, model)

    elif new_day == "New day":
        logger.info("Revising Idenity")
        new_currently, new_daily_req = revise_identity(scratch, a_mem, model)
        scratch.currently = new_currently
        scratch.daily_plan_req = new_daily_req

    logger.info("Making hourly schedule")
    scratch.f_daily_schedule = make_hourly_schedule(scratch, model, wake_up_hour)
    scratch.f_daily_schedule_hourly_org = scratch.f_daily_schedule[:]

    logger.info("Generating thought plan")
    data = generate_thought_plan(scratch, model)
    a_mem.add_thought(**data)


def revise_identity(scratch: PersonaScratch, a_mem: AssociativeMemory, model: LLM_API):
    p_name = scratch.name
    focal_points = [
        f"{p_name}'s plan for {scratch.get_str_curr_date_str()}.",
        f"Important recent events for {p_name}'s life.",
    ]

    retrieved = retrieve_focal_points(scratch, a_mem, focal_points, model)
    _, _, new_currently, new_daily_req = run_prompt_revise_identity(
        scratch, model, retrieved
    )
    return new_currently, new_daily_req


# FIXME try to make the code more readable instead of adding comments.
# also more performant, because this is really a bottleneck...
def make_hourly_schedule(scratch: PersonaScratch, model: LLM_API, wake_up_hour):
    hour_str = [
        "00:00 AM",
        "01:00 AM",
        "02:00 AM",
        "03:00 AM",
        "04:00 AM",
        "05:00 AM",
        "06:00 AM",
        # "07:00 AM",
        # "08:00 AM",
        # "09:00 AM",
        # "10:00 AM",
        # "11:00 AM",
        # "12:00 PM",
        # "01:00 PM",
        # "02:00 PM",
        # "03:00 PM",
        # "04:00 PM",
        # "05:00 PM",
        # "06:00 PM",
        # "07:00 PM",
        # "08:00 PM",
        # "09:00 PM",
        # "10:00 PM",
        # "11:00 PM",
    ]
    n_m1_activity = []
    diversity_repeat_count = 3
    for _ in range(diversity_repeat_count):
        n_m1_activity_set = set(n_m1_activity)
        if len(n_m1_activity_set) < 5:
            n_m1_activity = []
            for _, curr_hour_str in enumerate(hour_str):
                if wake_up_hour > 0:
                    n_m1_activity += ["sleeping"]
                    wake_up_hour -= 1
                else:
                    n_m1_activity += [
                        generate_hourly_schedule(
                            scratch, curr_hour_str, n_m1_activity, hour_str, model
                        )
                    ]

    _n_m1_hourly_compressed = []
    prev = None
    prev_count = 0
    for i in n_m1_activity:
        if i != prev:
            prev_count = 1
            _n_m1_hourly_compressed += [[i, prev_count]]
            prev = i
        else:
            if _n_m1_hourly_compressed:
                _n_m1_hourly_compressed[-1][1] += 1

    n_m1_hourly_compressed = []
    for task, duration in _n_m1_hourly_compressed:
        n_m1_hourly_compressed += [[task, duration * 60]]

    return n_m1_hourly_compressed


def generate_thought_plan(scratch: PersonaScratch, model: LLM_API):
    thought = (
        f"This is {scratch.name}'s plan for {scratch.curr_time.strftime('%A %B %d')}:"
    )
    for i in scratch.daily_req:
        thought += f" {i},"
    thought = thought[:-1] + "."

    created = scratch.curr_time
    expiration = scratch.curr_time + datetime.timedelta(days=30)
    s, p, o = (scratch.name, "plan", scratch.curr_time.strftime("%A %B %d"))
    keywords = {"plan"}
    thought_poignancy = 5
    thought_embedding_pair = (thought, model.get_embedding(thought))
    return {
        "created": created,
        "expiration": expiration,
        "s": s,
        "p": p,
        "o": o,
        "description": thought,
        "keywords": keywords,
        "poignancy": thought_poignancy,
        "embedding_pair": thought_embedding_pair,
        "filling": None,
    }


def generate_wake_up_hour(scratch: PersonaScratch, model):
    return int(run_prompt_wake_up(scratch, model)[0])


def generate_first_daily_plan(
    scratch: PersonaScratch, wake_up_hour: int, model: LLM_API
):
    return run_prompt_daily_plan(scratch, wake_up_hour, model)[0]


def generate_hourly_schedule(
    scratch: PersonaScratch,
    curr_hour_str: str,
    n_activity: list[str],
    hour_str: list[str],
    model: LLM_API,
):
    return run_prompt_hourly_schedule(
        scratch, curr_hour_str, n_activity, hour_str, model
    )[0]


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms
    from LLM_Character.persona.persona import Persona

    person = Persona("BANDER", "filesave")
    modelc = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    _long_term_planning(person, "First day", model)
