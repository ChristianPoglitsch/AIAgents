from LLM_Character.messages_dataclass import AIMessages


class UserScratch:
    def __init__(self, name: str):
        self.name = name
        self.chat = AIMessages()
        self.start_time_chatting = None
