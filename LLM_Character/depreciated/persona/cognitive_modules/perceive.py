from LLM_Character.communication.incoming_messages import EventData, OneLocationData
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.persona.cognitive_modules.reflect import generate_poig_score
from LLM_Character.persona.memory_structures.associative_memory.\
    associative_memory import (
        AssociativeMemory,
    )
from LLM_Character.persona.memory_structures.associative_memory.concept_node import (
    ConceptNode,
)
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.memory_structures.spatial_memory import MemoryTree


def perceive(
    cscratch: PersonaScratch,
    camem: AssociativeMemory,
    csmem: MemoryTree,
    new_character_location: OneLocationData,
    perceived_events: list[EventData],
    model: LLM_API,
) -> list[ConceptNode]:
    # update spatial knowledge
    csmem.update_oloc(new_character_location)

    ret_events = []
    for p_event in perceived_events:
        s = p_event.action_event_subject
        p = p_event.action_event_predicate
        o = p_event.action_event_object
        desc = p_event.action_event_description
        if not p:
            # If the object is not present, then we default the event to
            # "idle".
            p = "is"
            o = "idle"
            desc = "idle"
        desc = f"{s.split(':')[-1]} is {desc}"
        p_event = (s, p, o)

        latest_events = camem.get_summarized_latest_events(cscratch.retention)
        if p_event not in latest_events:
            keywords = set()
            sub = p_event[0]
            obj = p_event[2]
            if ":" in p_event[0]:
                sub = p_event[0].split(":")[-1]
            if ":" in p_event[2]:
                obj = p_event[2].split(":")[-1]
            keywords.update([sub, obj])

            desc_embedding_in = desc
            if "(" in desc:
                desc_embedding_in = (
                    desc_embedding_in.split("(")[1].split(")")[0].strip()
                )

            if desc_embedding_in in camem.embeddings:
                event_embedding = camem.embeddings[desc_embedding_in]
            else:
                event_embedding = model.get_embedding(desc_embedding_in)
            event_embedding_pair = (desc_embedding_in, event_embedding)

            # NOTE: different poig score for thoughts, convo and events? or
            # same function?
            event_poignancy = generate_poig_score(
                cscratch, desc_embedding_in, model)

            chat_node_ids = []

            # NOTE: dont think i need this here anymore,
            # since i'll do it in interact or in converse ? idk
            # you never observe your own chats....,
            # but then you'll never have chat_node_ids filled.
            # in the first place, why chats included in events ??? events dont
            # need associated chats?? idk

            # if p_event[0] == f"{persona.name}" and p_event[1] == "chat with":
            #   curr_event = persona.scratch.act_event
            #   if persona.scratch.act_description in persona.a_mem.embeddings:
            #     chat_embedding = persona.a_mem.embeddings[
            #                        persona.scratch.act_description]
            #   else:
            #     chat_embedding = get_embedding(persona.scratch
            #                                           .act_description)
            #   chat_embedding_pair = (persona.scratch.act_description,
            #                          chat_embedding)
            #   chat_poignancy = generate_poig_score(persona, "chat",
            #                                        persona.scratch.act_description)
            #   chat_node = persona.a_mem.add_chat(persona.scratch.curr_time, None,
            #                 curr_event[0], curr_event[1], curr_event[2],
            #                 persona.scratch.act_description, keywords,
            #                 chat_poignancy, chat_embedding_pair,
            #                 persona.scratch.chat)
            #   chat_node_ids = [chat_node.node_id]

            ret_events += [
                camem.add_event(
                    cscratch.curr_time,
                    None,
                    s,
                    p,
                    o,
                    desc,
                    keywords,
                    event_poignancy,
                    event_embedding_pair,
                    chat_node_ids,
                )
            ]
            cscratch.importance_trigger_curr -= event_poignancy
            cscratch.importance_ele_n += 1

    return ret_events
