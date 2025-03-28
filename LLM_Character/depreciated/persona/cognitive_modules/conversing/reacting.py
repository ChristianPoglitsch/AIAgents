from typing import Union

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage
from LLM_Character.persona.cognitive_modules.retrieve import retrieve_focal_points
from LLM_Character.persona.memory_structures.associative_memory.associative_memory import (
    AssociativeMemory,
    ConceptNode,
)
from LLM_Character.persona.memory_structures.scratch.persona_scratch import (
    PersonaScratch,
)
from LLM_Character.persona.memory_structures.scratch.user_scratch import UserScratch
from LLM_Character.persona.prompt_modules.converse_prompts.generate_iterative_chat import (
    run_prompt_iterative_chat,
)
from LLM_Character.persona.prompt_modules.converse_prompts.summarize_relationship import (
    run_prompt_summarize_relationship,
)


def _generate_response(
    user_scratch: UserScratch,
    character_scratch: PersonaScratch,
    character_mem: AssociativeMemory,
    message: str,
    model: LLM_API,
):
    # YOU WANT TO CREATE A MESSAGE DIRECTED TO USER!
    focal_points = [f"{user_scratch.name}"]
    retrieved = retrieve_focal_points(
        character_scratch, character_mem, focal_points, model, 50
    )
    relationship = generate_summarize_agent_relationship(
        user_scratch, character_scratch, model, retrieved
    )

    curr_chat = user_scratch.chat.get_messages()
    last_chat = ""
    for i in curr_chat[-4:]:
        last_chat += i.print_message_sender()

    if last_chat:
        focal_points = [
            f"{relationship}",
            f"{character_scratch.name} is {character_scratch.act_description}",
            last_chat,
        ]
    else:
        focal_points = [
            f"{relationship}",
            f"{character_scratch.name} is {character_scratch.act_description}",
        ]

    # NOTE add message as focal point
    # focal_points = [f"{user_scratch.name}", message]
    #retrieved = retrieve_focal_points(
    #    character_scratch, character_mem, focal_points, model, 15
    #)
    utt, emotion, trust, end = generate_one_utterance_user(
        user_scratch, character_scratch, character_mem, model, retrieved, curr_chat[-8:]
    )
    return utt, emotion, trust, end


def generate_summarize_agent_relationship(
    iscratch: Union[UserScratch, PersonaScratch],
    tscratch: PersonaScratch,
    model: LLM_API,
    retrieved: dict[str, list[ConceptNode]],
):
    all_embedding_keys = list()
    for _, val in retrieved.items():
        for i in val:
            all_embedding_keys += [i.embedding_key]
    all_embedding_key_str = ""
    for i in all_embedding_keys:
        all_embedding_key_str += f"{i}\n"

    if all_embedding_key_str == "":
        all_embedding_key_str = "There is no statements to share."

    summarized_relationship = run_prompt_summarize_relationship(
        iscratch, tscratch, model, all_embedding_key_str
    )[0]
    return summarized_relationship


def generate_one_utterance_user(
    uscratch: UserScratch,
    cscratch: PersonaScratch,
    camem: AssociativeMemory,
    model: LLM_API,
    retrieved: dict[str, list[ConceptNode]],
    curr_chat: list[AIMessage],
):
    # FIXME dont know if this context will always work, check later when
    # everything is in place.
    curr_context = (
        f"{cscratch.name} "
        + f"was {cscratch.act_description} "
        + f"when {cscratch.name} "
        + f"saw {uscratch.name} "
        +
        # FIXME for now, the user doesnt have any plans.
        # f"in the middle of {uscratch.act_description}.\n")
        ".\n"
    )
    curr_context += (
        f"{cscratch.name} " + "is initiating a conversation with " + f"{uscratch.name}."
    )
    x, y = run_prompt_iterative_chat(
        uscratch, cscratch, camem, model, retrieved, curr_context, curr_chat
    )
    return x["utterance"], x["emotion"], x["trust"], y["end"]


if __name__ == "__main__":
    from LLM_Character.llm_comms.llm_local import LocalComms

    from LLM_Character.persona.persona import Persona
    from LLM_Character.persona.user import User

    person = Persona("MIKE")
    user = User("MIKE")
    modelc = LocalComms()

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    modelc.init(model_id)

    model = LLM_API(modelc)
    message = "hi"
    utt, emotion, trust, end = _generate_response(user.scratch, person.scratch, person.a_mem, message, model)
    print(utt)
