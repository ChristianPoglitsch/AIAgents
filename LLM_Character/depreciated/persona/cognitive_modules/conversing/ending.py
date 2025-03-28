import datetime

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.memory_structures.scratch.user_scratch import UserScratch
from LLM_Character.persona.prompt_modules.converse_prompts.decomp_schedule import (
    run_prompt_decomp_schedule,
)
from LLM_Character.persona.prompt_modules.converse_prompts.summarize_conversation import (
    run_prompt_summarize_conversation,
)


def _end_conversation(
    user_scratch: UserScratch, character_scratch: PersonaScratch, model: LLM_API
):
    convo_summary = generate_convo_summary(user_scratch.chat.prints_messages_sender(), model)

    inserted_act = convo_summary
    inserted_act_dur = character_scratch.curr_time - user_scratch.start_time_chatting

    act_address = f"<persona> {user_scratch.name}"
    act_event = (character_scratch.name, "chat with", user_scratch.name)
    chatting_with = user_scratch.name
    chatting_with_buffer = {}
    chatting_with_buffer[user_scratch.name] = 800

    chatting_end_time = character_scratch.curr_time
    act_start_time = user_scratch.start_time_chatting

    _create_react(
        character_scratch,
        model,
        inserted_act,
        inserted_act_dur,
        act_address,
        act_event,
        chatting_with,
        user_scratch.chat,
        chatting_with_buffer,
        chatting_end_time,
        act_start_time,
    )


def generate_convo_summary(convo, model):
    convo_summary = run_prompt_summarize_conversation(model, convo)[0]
    return convo_summary


def _create_react(
    cscratch: PersonaScratch,
    model: LLM_API,
    inserted_act,
    inserted_act_dur,
    act_address,
    act_event,
    chatting_with,
    chat,
    chatting_with_buffer,
    chatting_end_time,
    act_start_time=None,
):
    min_sum = 0
    for i in range(cscratch.get_f_daily_schedule_hourly_org_index()):
        min_sum += cscratch.f_daily_schedule_hourly_org[i][1]
    start_hour = int(min_sum / 60)

    if cscratch.f_daily_schedule_hourly_org:  # NOTE ibr: added
        if (
            cscratch.f_daily_schedule_hourly_org[
                cscratch.get_f_daily_schedule_hourly_org_index()
            ][1]
            >= 120
        ):
            end_hour = (
                start_hour
                + cscratch.f_daily_schedule_hourly_org[
                    cscratch.get_f_daily_schedule_hourly_org_index()
                ][1]
                / 60
            )

        elif (
            cscratch.f_daily_schedule_hourly_org[
                cscratch.get_f_daily_schedule_hourly_org_index()
            ][1]
            + cscratch.f_daily_schedule_hourly_org[
                cscratch.get_f_daily_schedule_hourly_org_index() + 1
            ][1]
        ):
            end_hour = start_hour + (
                (
                    cscratch.f_daily_schedule_hourly_org[
                        cscratch.get_f_daily_schedule_hourly_org_index()
                    ][1]
                    + cscratch.f_daily_schedule_hourly_org[
                        cscratch.get_f_daily_schedule_hourly_org_index() + 1
                    ][1]
                )
                / 60
            )

        else:
            end_hour = start_hour + 2
        end_hour = int(end_hour)

        dur_sum = 0
        count = 0
        start_index = None
        end_index = None
        for _, dur in cscratch.f_daily_schedule:
            if dur_sum >= start_hour * 60 and start_index is None:
                start_index = count
            if dur_sum >= end_hour * 60 and end_index is None:
                end_index = count
            dur_sum += dur
            count += 1

        ret = generate_new_decomp_schedule(
            cscratch, model, inserted_act, inserted_act_dur, start_hour, end_hour
        )
        cscratch.f_daily_schedule[start_index:end_index] = ret

    cscratch.add_new_action(
        act_address,
        inserted_act_dur,
        inserted_act,
        act_event,
        chatting_with,
        chat,
        chatting_with_buffer,
        chatting_end_time,
        act_start_time,
    )


def generate_new_decomp_schedule(
    cscratch, model: LLM_API, inserted_act, inserted_act_dur, start_hour, end_hour
):
    today_min_pass = (
        int(cscratch.curr_time.hour) * 60 + int(cscratch.curr_time.minute) + 1
    )
    main_act_dur = []
    truncated_act_dur = []
    dur_sum = 0
    count = 0
    truncated_fin = False

    for act, dur in cscratch.f_daily_schedule:
        if (dur_sum >= start_hour * 60) and (dur_sum < end_hour * 60):
            main_act_dur += [[act, dur]]
            if dur_sum <= today_min_pass:
                truncated_act_dur += [[act, dur]]
            elif dur_sum > today_min_pass and not truncated_fin:
                truncated_act_dur += [
                    [cscratch.f_daily_schedule[count][0], dur_sum - today_min_pass]
                ]
                truncated_act_dur[-1][-1] -= dur_sum - today_min_pass
                truncated_fin = True
        dur_sum += dur
        count += 1

    main_act_dur = main_act_dur
    y = " (on the way to " + truncated_act_dur[-1][0].split("(")[-1][:-1] + ")"
    x = truncated_act_dur[-1][0].split("(")[0].strip() + y
    truncated_act_dur[-1][0] = x

    if "(" in truncated_act_dur[-1][0]:
        inserted_act = (
            truncated_act_dur[-1][0].split("(")[0].strip() + " (" + inserted_act + ")"
        )

    truncated_act_dur += [[inserted_act, inserted_act_dur]]
    start_time_hour = datetime.datetime(2022, 10, 31, 0, 0) + datetime.timedelta(
        hours=start_hour
    )
    end_time_hour = datetime.datetime(2022, 10, 31, 0, 0) + datetime.timedelta(
        hours=end_hour
    )

    return run_prompt_decomp_schedule(
        cscratch,
        main_act_dur,
        truncated_act_dur,
        start_time_hour,
        end_time_hour,
        inserted_act,
        inserted_act_dur,
        model,
    )[0]


if __name__ == "__main__":
    from llm_comms.llm_local import LocalComms

    from LLM_Character.persona.persona import Persona
    from LLM_Character.persona.user import User

    person = Persona("MIKE", "nice")
    user = User("MIKE", "nice")
    modelc = LocalComms()

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    message = "hi"
    _end_conversation(user.scratch, person.scratch, model)
