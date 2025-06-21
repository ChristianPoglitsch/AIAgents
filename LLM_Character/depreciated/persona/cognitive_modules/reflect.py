import datetime

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.persona.cognitive_modules.converse import generate_event_poig_score
from LLM_Character.persona.cognitive_modules.retrieve import retrieve_focal_points
from LLM_Character.persona.memory_structures.associative_memory.\
    associative_memory import (
        AssociativeMemory,
    )
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.prompt_modules.converse_prompts.poignancy_chat import (
    run_prompt_poignancy_chat,
)
from LLM_Character.persona.prompt_modules.planning_prompts.determine_action.\
    event_triple import (
        run_prompt_event_triple,
    )
from LLM_Character.persona.prompt_modules.reflect_prompts.\
    planning_thought_convo import (
        run_prompt_planning_thought_convo,
    )
from LLM_Character.persona.prompt_modules.reflect_prompts.focal_pt import (
    run_prompt_generate_focal_pt,
)
from LLM_Character.persona.prompt_modules.reflect_prompts.insight_and_guidance import (
    run_prompt_insight_and_evidence,
)
from LLM_Character.persona.prompt_modules.reflect_prompts.memo_convo import (
    run_prompt_memo_convo,
)

# FIXME: where reset scratch_chatting_end_time ? normally in maze or
# reverie no?


def reflect(scratch: PersonaScratch, a_mem: AssociativeMemory, model: LLM_API):
    if reflection_trigger(scratch, a_mem):
        run_reflect(scratch, a_mem, model)
        reset_reflection_counter(scratch)

    if scratch.chatting_end_time:
        if scratch.curr_time + datetime.timedelta(0, 10) == scratch.chatting_end_time:
            all_utt = scratch.chat.prints_messages_sender()
            last_chat = a_mem.get_last_chat(scratch.chatting_with)
            evidence = [last_chat.node_id]

            planning_thought = generate_planning_thought_on_convo(
                scratch, all_utt, model
            )
            planning_thought = f"For {scratch.name}'s planning: {planning_thought}"

            created = scratch.curr_time
            expiration = scratch.curr_time + datetime.timedelta(days=30)
            s, p, o = generate_action_event_triple(
                scratch, model, planning_thought)
            keywords = {s, p, o}
            thought_poignancy = generate_event_poig_score(
                scratch, planning_thought, model
            )
            thought_embedding_pair = (
                planning_thought,
                model.get_embedding(planning_thought),
            )

            a_mem.add_thought(
                created,
                expiration,
                s,
                p,
                o,
                planning_thought,
                keywords,
                thought_poignancy,
                thought_embedding_pair,
                evidence,
            )

            memo_thought = generate_memo_on_convo(scratch, all_utt, model)
            memo_thought = f"{scratch.name} {memo_thought}"

            created = scratch.curr_time
            expiration = scratch.curr_time + datetime.timedelta(days=30)
            s, p, o = generate_action_event_triple(
                scratch, model, planning_thought)
            keywords = {s, p, o}
            thought_poignancy = generate_event_poig_score(
                scratch, memo_thought, model)
            thought_embedding_pair = (
                memo_thought, model.get_embedding(memo_thought))

            a_mem.add_thought(
                created,
                expiration,
                s,
                p,
                o,
                memo_thought,
                keywords,
                thought_poignancy,
                thought_embedding_pair,
                evidence,
            )


def reflection_trigger(scratch: PersonaScratch, a_mem: AssociativeMemory):
    if (
        scratch.importance_trigger_curr <= 0
        and [] != a_mem.seq_event + a_mem.seq_thought
    ):
        return True
    return False


def run_reflect(scratch: PersonaScratch, a_mem: AssociativeMemory, model: LLM_API):
    focal_points = generate_focal_points(scratch, a_mem, model, 3)
    retrieved = retrieve_focal_points(scratch, a_mem, focal_points, model)

    for _, nodes in retrieved.items():
        thoughts = generate_insights_and_evidence(nodes, model, 5)
        for thought, evidence in thoughts.items():
            created = scratch.curr_time
            expiration = scratch.curr_time + datetime.timedelta(days=30)
            s, p, o = generate_action_event_triple(scratch, model, thought)
            keywords = {s, p, o}
            thought_poignancy = generate_poig_score(scratch, thought, model)
            thought_embedding_pair = (thought, model.get_embedding(thought))

            a_mem.add_thought(
                created,
                expiration,
                s,
                p,
                o,
                thought,
                keywords,
                thought_poignancy,
                thought_embedding_pair,
                evidence,
            )


def generate_focal_points(
    scratch: PersonaScratch, a_mem: AssociativeMemory, model: LLM_API, n=3
):
    nodes = [
        [i.last_accessed, i]
        for i in a_mem.seq_thought + a_mem.seq_event
        if "idle" not in i.embedding_key
    ]

    nodes = sorted(nodes, key=lambda x: x[0])
    nodes = [i for _, i in nodes]

    statements = ""
    for node in nodes[-1 * scratch.importance_ele_n:]:
        statements += node.embedding_key + "\n"
    return run_prompt_generate_focal_pt(model, n, statements)[0]


def generate_insights_and_evidence(nodes, model: LLM_API, n=5):
    statements = ""
    for count, node in enumerate(nodes):
        statements += f"{str(count)}. {node.embedding_key}\n"

    ret = run_prompt_insight_and_evidence(model, n, statements)[0]

    try:
        for thought, evi_raw in ret.items():
            evidence_node_id = [nodes[i].node_id for i in evi_raw]
            ret[thought] = evidence_node_id
        return ret
    except BaseException:
        return {"this is blank": "node_1"}


def reset_reflection_counter(scratch):
    persona_imt_max = scratch.importance_trigger_max
    scratch.importance_trigger_curr = persona_imt_max
    scratch.importance_ele_n = 0


def generate_planning_thought_on_convo(scratch, all_utt, model: LLM_API):
    return run_prompt_planning_thought_convo(scratch, model, all_utt)[0]


def generate_poig_score(scratch: PersonaScratch, description: str, model: LLM_API):
    return run_prompt_poignancy_chat(scratch, description, model)[0]


def generate_memo_on_convo(scratch: PersonaScratch, all_utt: str, model: LLM_API):
    return run_prompt_memo_convo(scratch, model, all_utt)[0]


def generate_action_event_triple(scratch, model, act_desp):
    return run_prompt_event_triple(scratch, model, act_desp)[0]
