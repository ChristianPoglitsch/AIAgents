import datetime
import json
import os
from typing import Union

from LLM_Character.depreciated.persona.memory_structures.associative_memory.concept_node import (
    ConceptNode,
)


class AssociativeMemory:
    def __init__(self):
        self.id_to_node: dict[str, ConceptNode] = {}

        self.seq_event: list[ConceptNode] = []
        self.seq_thought: list[ConceptNode] = []
        self.seq_chat: list[ConceptNode] = []

        self.kw_to_event: dict[str, list[ConceptNode]] = {}
        self.kw_to_thought: dict[str, list[ConceptNode]] = {}
        self.kw_to_chat: dict[str, list[ConceptNode]] = {}

        self.kw_strength_event: dict[str, int] = {}
        self.kw_strength_thought: dict[str, int] = {}
        self.embeddings: dict[str, list[float]] = {}

    def load_from_file(self, f_saved: str):
        self.load(f_saved)

    def add_thought(
        self,
        created,
        expiration,
        s,
        p,
        o,
        description,
        keywords,
        poignancy,
        embedding_pair,
        filling,
    ):
        # Setting up the node ID and counts.
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.seq_thought) + 1
        node_type = "thought"
        node_id = f"node_{str(node_count)}"
        depth = 1
        try:
            if filling:
                depth += max([self.id_to_node[i].depth for i in filling])
        except BaseException:
            pass

        # Creating the <ConceptNode> object.
        node = ConceptNode(
            node_id,
            node_count,
            type_count,
            node_type,
            depth,
            created,
            expiration,
            s,
            p,
            o,
            description,
            embedding_pair[0],
            poignancy,
            keywords,
            filling,
        )

        # Creating various dictionary cache for fast access.
        self.seq_thought[0:0] = [node]
        keywords = [i.lower() for i in keywords]
        for kw in keywords:
            if kw in self.kw_to_thought:
                self.kw_to_thought[kw][0:0] = [node]
            else:
                self.kw_to_thought[kw] = [node]
        self.id_to_node[node_id] = node

        # Adding in the kw_strength
        if f"{p} {o}" != "is idle":
            for kw in keywords:
                if kw in self.kw_strength_thought:
                    self.kw_strength_thought[kw] += 1
                else:
                    self.kw_strength_thought[kw] = 1

        self.embeddings[embedding_pair[0]] = embedding_pair[1]

        return node

    def add_chat(
        self,
        created,
        expiration,
        s,
        p,
        o,
        description,
        keywords,
        poignancy,
        embedding_pair,
        filling,
    ):
        # Setting up the node ID and counts.
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.seq_chat) + 1
        node_type = "chat"
        node_id = f"node_{str(node_count)}"
        depth = 0

        # Creating the <ConceptNode> object.
        node = ConceptNode(
            node_id,
            node_count,
            type_count,
            node_type,
            depth,
            created,
            expiration,
            s,
            p,
            o,
            description,
            embedding_pair[0],
            poignancy,
            keywords,
            filling,
        )

        # Creating various dictionary cache for fast access.
        self.seq_chat[0:0] = [node]
        keywords = [i.lower() for i in keywords]
        for kw in keywords:
            if kw in self.kw_to_chat:
                self.kw_to_chat[kw][0:0] = [node]
            else:
                self.kw_to_chat[kw] = [node]
        self.id_to_node[node_id] = node

        self.embeddings[embedding_pair[0]] = embedding_pair[1]

        return node

    def add_event(
        self,
        created,
        expiration,
        s,
        p,
        o,
        description,
        keywords,
        poignancy,
        embedding_pair,
        filling,
    ):
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.seq_event) + 1
        node_type = "event"
        node_id = f"node_{str(node_count)}"
        depth = 0

        if "(" in description:
            description = (
                " ".join(description.split()[:3])
                + " "
                + description.split("(")[-1][:-1]
            )

        node = ConceptNode(
            node_id,
            node_count,
            type_count,
            node_type,
            depth,
            created,
            expiration,
            s,
            p,
            o,
            description,
            embedding_pair[0],
            poignancy,
            keywords,
            filling,
        )

        self.seq_event[0:0] = [node]
        keywords = [i.lower() for i in keywords]
        for kw in keywords:
            if kw in self.kw_to_event:
                self.kw_to_event[kw][0:0] = [node]
            else:
                self.kw_to_event[kw] = [node]
        self.id_to_node[node_id] = node

        if f"{p} {o}" != "is idle":
            for kw in keywords:
                if kw in self.kw_strength_event:
                    self.kw_strength_event[kw] += 1
                else:
                    self.kw_strength_event[kw] = 1

        self.embeddings[embedding_pair[0]] = embedding_pair[1]

        return node

    def retrieve_relevant_thoughts(
        self, s_content, p_content, o_content
    ) -> set[ConceptNode]:
        contents = [s_content, p_content, o_content]

        ret = []
        for content in contents:
            if content in self.kw_to_thought:
                ret += self.kw_to_thought[content.lower()]

        ret = set(ret)
        return ret

    def retrieve_relevant_events(
        self, s_content, p_content, o_content
    ) -> set[ConceptNode]:
        contents = [s_content, p_content, o_content]

        ret = []
        for i in contents:
            if i in self.kw_to_event:
                ret += self.kw_to_event[i]

        ret = set(ret)
        return ret

    def get_str_seq_thoughts(self):
        ret_str = ""
        for count, event in enumerate(self.seq_thought):
            ret_str += (
                f'{"Thought"} {len(self.seq_thought) - count}: '
                f'{event.spo_summary()} -- {event.description}'
            )
        return ret_str

    def get_str_seq_chats(self):
        ret_str = ""
        for _, event in enumerate(self.seq_chat):
            ret_str += f"Chat with {event.object}\n"
            ret_str += f'{event.created.strftime("%B %d, %Y, %H:%M:%S")}\n'
            for row in event.filling:
                ret_str += f"{row[0]}: {row[1]}\n"
        return ret_str

    def get_last_chat(
        self, target_persona_name: Union[str, None]
    ) -> Union[ConceptNode, None]:
        if target_persona_name and target_persona_name.lower() in self.kw_to_chat:
            return self.kw_to_chat[target_persona_name.lower()][0]
        return None

    def get_summarized_latest_events(self, retention):
        ret_set = set()
        for e_node in self.seq_event[:retention]:
            ret_set.add(e_node.spo_summary())
        return ret_set

    def load(self, f_saved: str):
        # TODO check if file exists ...

        self.embeddings: dict[str, list[float]] = json.load(
            open(f_saved + "/embeddings.json")
        )

        nodes_load = json.load(open(f_saved + "/nodes.json"))
        for count in range(len(nodes_load.keys())):
            node_id = f"node_{str(count+1)}"
            node_details = nodes_load[node_id]

            # NOTE is not used?
            # node_count = node_details["node_count"]
            # type_count = node_details["type_count"]
            node_type = node_details["type"]
            # depth = node_details["depth"]

            created = datetime.datetime.strptime(
                node_details["created"], "%Y-%m-%d %H:%M:%S"
            )
            expiration = None
            if node_details["expiration"]:
                expiration = datetime.datetime.strptime(
                    node_details["expiration"], "%Y-%m-%d %H:%M:%S"
                )

            s = node_details["subject"]
            p = node_details["predicate"]
            o = node_details["object"]

            description = node_details["description"]
            embedding_pair = (
                node_details["embedding_key"],
                self.embeddings[node_details["embedding_key"]],
            )
            poignancy = node_details["poignancy"]
            keywords = set(node_details["keywords"])
            filling = node_details["filling"]

            if node_type == "chat":
                self.add_chat(
                    created,
                    expiration,
                    s,
                    p,
                    o,
                    description,
                    keywords,
                    poignancy,
                    embedding_pair,
                    filling,
                )
            elif node_type == "thought":
                self.add_thought(
                    created,
                    expiration,
                    s,
                    p,
                    o,
                    description,
                    keywords,
                    poignancy,
                    embedding_pair,
                    filling,
                )

        kw_strength_load = json.load(open(f_saved + "/kw_strength.json"))
        if kw_strength_load["kw_strength_event"]:
            self.kw_strength_event = kw_strength_load["kw_strength_event"]
        if kw_strength_load["kw_strength_thought"]:
            self.kw_strength_thought = kw_strength_load["kw_strength_thought"]

    def save(self, out_json):
        os.makedirs(os.path.dirname(out_json), exist_ok=True)

        r = {}
        for count in range(len(self.id_to_node.keys()), 0, -1):
            node_id = f"node_{str(count)}"
            node = self.id_to_node[node_id]

            r[node_id] = {}
            r[node_id]["node_count"] = node.node_count
            r[node_id]["type_count"] = node.type_count
            r[node_id]["type"] = node.node_type
            r[node_id]["depth"] = node.depth

            r[node_id]["created"] = node.created.strftime("%Y-%m-%d %H:%M:%S")
            r[node_id]["expiration"] = None
            if node.expiration:
                r[node_id]["expiration"] = node.expiration.strftime("%Y-%m-%d %H:%M:%S")

            r[node_id]["subject"] = node.subject
            r[node_id]["predicate"] = node.predicate
            r[node_id]["object"] = node.object

            r[node_id]["description"] = node.description
            r[node_id]["embedding_key"] = node.embedding_key
            r[node_id]["poignancy"] = node.poignancy
            r[node_id]["keywords"] = list(node.keywords)
            r[node_id]["filling"] = node.filling

        with open(out_json + "nodes.json", "w") as outfile:
            json.dump(r, outfile)

        r = {}
        r["kw_strength_event"] = self.kw_strength_event
        r["kw_strength_thought"] = self.kw_strength_thought
        with open(out_json + "kw_strength.json", "w") as outfile:
            json.dump(r, outfile)

        with open(out_json + "embeddings.json", "w") as outfile:
            json.dump(self.embeddings, outfile)
