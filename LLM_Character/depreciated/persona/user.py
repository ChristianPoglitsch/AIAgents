from LLM_Character.persona.cognitive_modules.converse import chatting
from LLM_Character.persona.memory_structures.scratch.user_scratch import UserScratch

# User represents the human who plays the game.


class User:
    scratch: UserScratch

    def __init__(self, name: str):
        self.scratch = UserScratch(name)

    def move(self, curr_location):
        self.scratch.curr_location = curr_location

    def open_convo_session(self, data, model) -> str:
        return chatting(self.scratch, data, model)
