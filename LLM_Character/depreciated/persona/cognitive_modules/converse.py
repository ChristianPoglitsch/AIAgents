import logging
from typing import Tuple

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.persona.cognitive_modules.conversing.ending import _end_conversation
from LLM_Character.persona.cognitive_modules.conversing.reacting import (
    _generate_response,
)
from LLM_Character.persona.memory_structures.associative_memory.associative_memory import (  # noqa: E501
    AssociativeMemory,
)
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.memory_structures.scratch.user_scratch import UserScratch
from LLM_Character.persona.prompt_modules.converse_prompts.poignancy_chat import (
    run_prompt_poignancy_chat,
)
from LLM_Character.persona.prompt_modules.converse_prompts.poignancy_event import (
    run_prompt_poignancy_event,
)
from LLM_Character.util import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def chatting(
    user_scratch: UserScratch,
    character_scratch: PersonaScratch,
    character_mem: AssociativeMemory,
    message: str,
    model: LLM_API,
) -> Tuple[str, str, int, bool]:
    if len(user_scratch.chat) == 0:
        user_scratch.start_time_chatting = character_scratch.curr_time

    ai_message = AIMessage(message=message, role="user", class_type="MessageAI", sender=user_scratch.name)
    user_scratch.chat.add_message(ai_message)

    logger.info("generating response")
    utt, emotion, trust, end = _generate_response(
        user_scratch, character_scratch, character_mem, message, model
    )

    ai_message = AIMessage(message=utt, role="user", class_type="MessageAI", sender=character_scratch.name)
    user_scratch.chat.add_message(ai_message)

    character_scratch.curr_emotion = emotion.lower()
    character_scratch.curr_trust[user_scratch.name] = int(trust)

    if end:
        logger.info("ending conversation")
        _end_conversation(user_scratch, character_scratch, model)
        p_event = character_scratch.get_curr_event_and_desc()

        logger.info("remembering chat")
        nodeid, keywords = _remember_chat(
            character_scratch, character_mem, p_event, model
        )
        _prepare_reflect_chat(
            character_scratch, character_mem, p_event, keywords, [nodeid], model
        )

        user_scratch.chat = AIMessages()
        user_scratch.start_time_chatting = None

        # reset
        character_scratch.chatting_with = None
        character_scratch.chat = AIMessages()
        character_scratch.chatting_with_buffer = {}
        character_scratch.chatting_end_time = None

    return utt, emotion.lower(), trust, end


def _remember_chat(
    cscratch: PersonaScratch, ca_mem: AssociativeMemory, p_event, model: LLM_API
):
    s, p, o, desc = p_event
    desc = f"{s.split(':')[-1]} is {desc}"
    p_event = (s, p, o)

    keywords = set()
    sub = p_event[0]
    obj = p_event[2]
    if ":" in p_event[0]:
        sub = p_event[0].split(":")[-1]
    if ":" in p_event[2]:
        obj = p_event[2].split(":")[-1]
    keywords.update([sub, obj])

    if p_event[0] == f"{cscratch.name}" and p_event[1] == "chat with":
        curr_event = cscratch.act_event

        if cscratch.act_description in ca_mem.embeddings:
            chat_embedding = ca_mem.embeddings[cscratch.act_description]
        else:
            chat_embedding = model.get_embedding(cscratch.act_description)

        chat_embedding_pair = (cscratch.act_description, chat_embedding)
        chat_poignancy = generate_chat_poig_score(cscratch, model)
        node = ca_mem.add_chat(
            cscratch.curr_time,
            None,
            curr_event[0],
            curr_event[1],
            curr_event[2],
            cscratch.act_description,
            keywords,
            chat_poignancy,
            chat_embedding_pair,
            cscratch.chat.get_sender_message_format(),
        )

        # FIXME: add this to a_mem with add_event.
        return node.node_id, keywords


def generate_chat_poig_score(scratch: PersonaScratch, model: LLM_API):
    return run_prompt_poignancy_chat(scratch, scratch.act_description, model)[0]


def _prepare_reflect_chat(
    scratch: PersonaScratch,
    a_mem: AssociativeMemory,
    event: Tuple[str, str, str, str],
    keywords: set,
    chat_node_ids: list[str],
    model: LLM_API,
):
    s, p, o, desc = event
    desc_embedding_in = desc
    if "(" in desc:
        desc_embedding_in = desc_embedding_in.split("(")[1].split(")")[0].strip()
    if desc_embedding_in in a_mem.embeddings:
        event_embedding = a_mem.embeddings[desc_embedding_in]
    else:
        event_embedding = model.get_embedding(desc_embedding_in)
    event_embedding_pair = (desc_embedding_in, event_embedding)

    event_poignancy = generate_event_poig_score(scratch, desc_embedding_in, model)
    a_mem.add_event(
        scratch.curr_time,
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
    scratch.importance_trigger_curr -= event_poignancy
    scratch.importance_ele_n += 1


def generate_event_poig_score(
    scratch: PersonaScratch, description: str, model: LLM_API
):
    return run_prompt_poignancy_event(scratch, description, model)[0]
