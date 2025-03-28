from sentence_transformers import util

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.persona.memory_structures.associative_memory.associative_memory import (  # noqa: E501
    AssociativeMemory,
    ConceptNode,
)
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)


def retrieve_recent_sorted_nodes(a_mem: AssociativeMemory):
    nodes = []
    for i in a_mem.seq_thought + a_mem.seq_event + a_mem.seq_chat:
        if "idle" not in i.embedding_key:
            nodes += [[i.last_accessed, i]]

    nodes = sorted(nodes, key=lambda x: x[0])
    nodes = [i for _, i in nodes]
    return nodes


def extract_recency(scratch: PersonaScratch, nodes: list[ConceptNode]):
    recency_vals = [scratch.recency_decay**i for i in range(1, len(nodes) + 1)]

    recency_out = {}
    for count, node in enumerate(nodes):
        recency_out[node.node_id] = recency_vals[count]
    return recency_out


def extract_importance(nodes: list[ConceptNode]):
    importance_out = {}
    for _, node in enumerate(nodes):
        importance_out[node.node_id] = node.poignancy
    return importance_out


def extract_relevance(
    a_mem: AssociativeMemory, nodes: list[ConceptNode], focal_pt: str, model: LLM_API
):
    focal_embedding = model.get_embedding(focal_pt)
    relevance_out = {}
    for _, node in enumerate(nodes):
        node_embedding = a_mem.embeddings[node.embedding_key]
        relevance_out[node.node_id] = cos_sim(node_embedding, focal_embedding)
    return relevance_out


def cos_sim(a: list[float], b: list[float]):
    return util.pytorch_cos_sim(a, b)


def normalize_dict_floats(d: dict, target_min: float, target_max: float):
    if d:
        min_val = min(val for val in d.values())
        max_val = max(val for val in d.values())
        range_val = max_val - min_val

        if range_val == 0:
            for key, _ in d.items():
                d[key] = (target_max - target_min) / 2
        else:
            for key, val in d.items():
                d[key] = (val - min_val) * (
                    target_max - target_min
                ) / range_val + target_min
    return d


def top_highest_x_values(d: dict, x: int):
    top_v = dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:x])
    return top_v
