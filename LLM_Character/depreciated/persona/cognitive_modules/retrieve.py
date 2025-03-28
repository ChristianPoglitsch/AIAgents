from dataclasses import dataclass
from typing import Dict

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.persona.cognitive_modules.retrieving.util import (
    extract_importance,
    extract_recency,
    extract_relevance,
    normalize_dict_floats,
    retrieve_recent_sorted_nodes,
    top_highest_x_values,
)
from LLM_Character.persona.memory_structures.associative_memory.associative_memory import (  # noqa: E501
    AssociativeMemory,
    ConceptNode,
)
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)


@dataclass
class EventContext:
    curr_event: ConceptNode
    events: list[ConceptNode]
    thoughts: list[ConceptNode]

    def __str__(self):
        """Return a formatted string representation of the EventContext"""

        def format_list(lst):
            """Format a list of ConceptNode objects or None."""
            return ", ".join(str(item) for item in lst) if lst else "None"

        return (
            f"EventContext(\n"
            f"  curr_event={self.curr_event},\n"
            f"  events=[{format_list(self.events)}],\n"
            f"  thoughts=[{format_list(self.thoughts)}]\n"
            f")"
        )


def retrieve_contextual_events(
    a_mem: AssociativeMemory, perceived: list[ConceptNode]
) -> Dict[str, EventContext]:
    retrieved = {}
    for event in perceived:
        relevant_events = a_mem.retrieve_relevant_events(
            event.subject, event.predicate, event.object
        )  # return set of ConceptNode

        relevant_thoughts = a_mem.retrieve_relevant_thoughts(
            event.subject, event.predicate, event.object
        )  # return set of ConceptNode

        context = EventContext(
            curr_event=event,
            events=list(relevant_events),
            thoughts=list(relevant_thoughts),
        )
        retrieved[event.description] = context

    return retrieved


def retrieve_focal_points(
    scratch: PersonaScratch,
    a_mem: AssociativeMemory,
    focal_points: list[str],
    model: LLM_API,
    n_count=30,
) -> dict[str, list[ConceptNode]]:
    retrieved = {}
    for focal_pt in focal_points:
        nodes = retrieve_recent_sorted_nodes(a_mem)

        recency_out = extract_recency(scratch, nodes)
        recency_out = normalize_dict_floats(recency_out, 0, 1)

        importance_out = extract_importance(nodes)
        importance_out = normalize_dict_floats(importance_out, 0, 1)

        #relevance_out = extract_relevance(a_mem, nodes, focal_pt, model)
        #relevance_out = normalize_dict_floats(relevance_out, 0, 1)

        gw = [0.5, 3, 2]
        master_out = {}
        for key in recency_out.keys():
            master_out[key] = (
                scratch.recency_w * recency_out[key] * gw[0]
                #+ scratch.relevance_w * relevance_out[key] * gw[1]
                + scratch.importance_w * importance_out[key] * gw[2]
            )

        master_out = top_highest_x_values(master_out, n_count)
        master_nodes = [a_mem.id_to_node[key] for key in list(master_out.keys())]

        for n in master_nodes:
            n.last_accessed = scratch.curr_time
        retrieved[focal_pt] = master_nodes
    return retrieved
