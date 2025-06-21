from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.memory_structures.spatial_memory import MemoryTree
from LLM_Character.persona.prompt_modules.planning_prompts.determine_action.action_arena import (  # noqa: E501
    run_prompt_action_arena,
)
from LLM_Character.persona.prompt_modules.planning_prompts.determine_action.action_game_object import (  # noqa: E501
    run_prompt_action_game_object,
)
from LLM_Character.persona.prompt_modules.planning_prompts.determine_action.action_sector import (  # noqa: E501
    run_prompt_action_sector,
)
from LLM_Character.persona.prompt_modules.planning_prompts.determine_action.event_triple import (  # noqa: E501
    run_prompt_event_triple,
)
from LLM_Character.persona.prompt_modules.planning_prompts.determine_action.task_decomp import (  # noqa: E501
    run_prompt_task_decomp,
)


def _determine_action(scratch: PersonaScratch, s_mem: MemoryTree, model: LLM_API):  # noqa: C901
    curr_index = scratch.get_f_daily_schedule_index()
    curr_index_60 = scratch.get_f_daily_schedule_index(advance=60)
    if curr_index == 0:
        act_desp, act_dura = scratch.f_daily_schedule[curr_index]
        if act_dura >= 60:
            if determine_decomp(act_desp, act_dura):
                scratch.f_daily_schedule[curr_index : curr_index + 1] = (
                    generate_task_decomp(scratch, model, act_desp, act_dura)
                )

        if curr_index_60 + 1 < len(scratch.f_daily_schedule):
            act_desp, act_dura = scratch.f_daily_schedule[curr_index_60 + 1]
            if act_dura >= 60:
                if determine_decomp(act_desp, act_dura):
                    scratch.f_daily_schedule[curr_index_60 + 1 : curr_index_60 + 2] = (
                        generate_task_decomp(scratch, model, act_desp, act_dura)
                    )

    if curr_index_60 < len(scratch.f_daily_schedule):
        if scratch.curr_time.hour < 23:
            act_desp, act_dura = scratch.f_daily_schedule[curr_index_60]
            if act_dura >= 60:
                if determine_decomp(act_desp, act_dura):
                    scratch.f_daily_schedule[curr_index_60 : curr_index_60 + 1] = (
                        generate_task_decomp(scratch, model, act_desp, act_dura)
                    )

    # 1440 is 24 * 60 , #minutes in a day. you want your daily schedule to
    # encompass your whole day.
    x_emergency = 0
    for i in scratch.f_daily_schedule:
        x_emergency += i[1]
    scratch.f_daily_schedule += [["sleeping", 1440 - x_emergency]]

    # print(x_emergency)
    # if 1440 - x_emergency > 0:
    #   print ("x_emergency__AAA", x_emergency)

    act_desp, act_dura = scratch.f_daily_schedule[curr_index]
    act_world = scratch.get_curr_location()["world"]
    act_sector = generate_action_sector(scratch, s_mem, model, act_desp)
    # print(act_sector)
    act_arena = generate_action_arena(
        scratch, s_mem, model, act_desp, act_world, act_sector
    )
    act_game_object = generate_action_game_object(
        scratch, s_mem, model, act_desp, act_world, act_sector, act_arena
    )
    act_event = generate_action_event_triple(scratch, model, act_desp)

    new_address = f"{act_world}:{act_sector}:{act_arena}:{act_game_object}"
    # FIXME: no chats here, normal?
    scratch.add_new_action(
        new_address, int(act_dura), act_desp, act_event, None, None, None, None
    )


def determine_decomp(act_desp, act_dura):
    if "sleep" not in act_desp and "bed" not in act_desp:
        return True
    elif "sleeping" in act_desp or "asleep" in act_desp or "in bed" in act_desp:
        return False
    elif "sleep" in act_desp or "bed" in act_desp:
        if act_dura > 60:
            return False
    return True


def generate_task_decomp(scratch: PersonaScratch, model: LLM_API, task, duration):
    return run_prompt_task_decomp(scratch, model, task, duration)[0]


def generate_action_sector(
    scratch: PersonaScratch, s_mem: MemoryTree, model: LLM_API, act_desp: str
):
    return run_prompt_action_sector(scratch, s_mem, model, act_desp)[0]


def generate_action_arena(
    scratch: PersonaScratch,
    s_mem: MemoryTree,
    model: LLM_API,
    act_desp: str,
    act_world: str,
    act_sector: str,
):
    return run_prompt_action_arena(
        scratch, s_mem, model, act_desp, act_world, act_sector
    )[0]


def generate_action_game_object(
    scratch: PersonaScratch,
    s_mem: MemoryTree,
    model: LLM_API,
    act_desp: str,
    act_world: str,
    act_sector: str,
    act_arena: str,
):
    if not s_mem.get_str_accessible_arena_game_objects(
        act_world, act_sector, act_arena
    ):
        return "<random>"
    return run_prompt_action_game_object(
        scratch, s_mem, model, act_desp, act_world, act_sector, act_arena
    )[0]


def generate_action_event_triple(
    scratch: PersonaScratch, model: LLM_API, act_desp: str
):
    return run_prompt_event_triple(scratch, model, act_desp)[0]


if __name__ == "__main__":
    from persona import Persona

    p = Persona("IBI")
    _determine_action(p)
