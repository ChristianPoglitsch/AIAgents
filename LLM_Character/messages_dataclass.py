"""Module providing dataclasses responsible for storing messages."""

import json
from typing import List


class AIMessage:
    """A single message of a chat."""

    def __init__(self,  message: str, sender: str = 'user', role : str ='user', class_type : str = 'MessageAI'):
        """
        Initializes the AIMessage instance.

        Args:
            message (str): The message sent in the chat.
            sender (str): Name of sender. Can be a user/usename/avatar name.
            role (str): The sender of the message, either `user` or `assistant`.
            class_type (str): Field `class_type` to indicate if this message to indicate if message is part of the system background for the LLM. (e.g. `MessageAI` or `Introduction`)
        """
        self.message = message
        self.role = role
        self.sender = sender
        self.class_type = class_type

    def get_formatted(self) -> dict[str, str]:
        """
        Formats the message as a dictionary with fields `class_type`, `role`, and `content`.

        Returns:
            dict[str, str]: A dictionary representing the message
            with `class_type`, `role`, and `content`.
        """
        return {
            "content": self.message,
            "role": self.role,
        }

    def print_message_role(self) -> str:
        """
        Formats the message as a string with the format `[role] content \\n`.
        Returns:
            str: The formatted message string.
        """
        return "[" + self.role + "] " + self.message + "\n"

    def print_message_sender(self) -> str:
        """
        Formats the message as a string with the format `sender: content \\n`.
        """
        return self.sender + ": " + self.message

    def get_user_message(self) -> str:
        """
        Returns the content of the message if the role is 'user'
        and the message is not part of the system background.
        Returns `None` otherwise.

        Returns:
            str or None: The content of the message if it is
            from the 'user' role and not part of the system background; otherwise, `None`.
        """
        if self.class_type == "MessageAI" and self.role == "user":
            return self.message
        return None

    def get_message(self) -> str:
        return self.message

    def get_role(self) -> str:
        """
        Returns the role of this message.

        Returns:
            str: The role of the message (e.g., 'user' or 'assistant').
        """
        return self.role

    def get_type(self) -> str:
        """
        Returns the `class_type` of this message,
        indicating if it is part of the system background or not.

        Returns:
            str: The class type of the message (e.g., 'MessageAI' or 'Introduction').
        """
        return self.class_type


class AIMessages:
    """
    Represents the messages of an entire chat.
    """

    def __init__(self):
        self.messages: List[AIMessage] = []

    def add_message(self, message: AIMessage):
        """
        Creates an AIMessage instance with the given `message` and `role`.

        Args:
            message (AIMessage): The content of the message.
        """        
        self.messages.append(message)

    def get_messages(self) -> List[AIMessage]:
        """
        Retrieves all the messages stored in this chat.

        Returns:
            List[AIMessage]: A list of AIMessage instances representing the chat history.
        """
        return self.messages

    def remove_item(self, index : int):
        """
        Remove item at index from list

        """
        self.messages.pop(index)

    def get_formatted(self) -> List[dict]:
        """
        Represents the messages of this chat as a list of
        dictionaries with fields `class_type`, `role`, and `content`.

        Returns:
            List[dict]: A list of dictionaries representing the formatted messages.
        """
        message_list = []
        for item in self.messages:
            message_list.append(item.get_formatted())
        return message_list

    def prints_messages_role(self):
        """
        Represents the messages of this chat as a single string,
        with each message formatted as `[role] content \\n`.

        Returns:
            str: A string containing all the messages formatted.
        """
        m = ""
        for item in self.messages:
            m = m + item.print_message_role()
        return m

    def prints_messages_sender(self):
        """
        Represents the messages of this chat as a single string,
        with each message formatted as `sender : content \\n`.

        Returns:
            str: A string containing all the messages formatted.
        """
        m = ""
        for item in self.messages:
            m = m + item.print_message_sender() + "\n"
        return m

    def get_user_message(self) -> AIMessage:
        """
        Concatenates all messages with the role 'user' into a single string.
        Returns an `AIMessage` object with the concatenated string as the content of the message.

        Returns:
            AIMessage: An AIMessage instance with the concatenated user messages.
        """
        user_message = ""
        for item in self.messages:
            m = item.get_user_message()
            if m is not None:
                user_message = user_message + m  # QUESTION: no delimter?
        return AIMessage(user_message, "user")

    def write_messages_to_json(self, file_path):
        """
        Writes the formatted messages to a JSON file.

        Args:
            file_path (str): The path to the file where the messages should be saved.
        """
        with open(file_path, "w") as json_file:
            json.dump(self.get_formatted(), json_file, indent=4)

    @staticmethod
    def read_messages_from_json(file_path):
        """
        Reads messages from a JSON file and populates a new AIMessages instance.

        Args:
            file_path (str): The path to the JSON file to read from.

        Returns:
            AIMessages: An AIMessages instance populated with the messages read from the file.
        """
        messages_ai = AIMessages()
        try:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                for item in data:
                    content = item.get("content")
                    role = item.get("role")
                    sender = item.get("sender")
                    class_type = item.get("class_type")
                    messages_ai.add_message(content, sender, role, class_type)
        finally:
            return messages_ai

    def get_sender_message_format(self) -> List[List[str]]:
        """
        Represents the messages of this chat as a list of lists,
        with each message containing only the `sender` and the `message`.

        Returns:
            List[List[str]]:
            A list of lists where each list is in the format [sender, message].
        """
        return [[item.sender, item.message] for item in self.messages]

    def __len__(self):
        return len(self.messages)


if __name__ == "__main__":
    from LLM_Character.communication.incoming_messages import PromptMessage

    # pm = PromptMessage(1, "Hallo Wereld")
    # print("JSON ", pm.toJSON())

    print("\n")
    print("--------------------")
    print("\n")

    ai_message = AIMessage(message="Hallo")
    print("message_formatted: ", ai_message.get_formatted())
    print("print_message: ", ai_message.print_message_role())
    print("get_user_message: ", ai_message.get_user_message())
    print("get_role: ", ai_message.get_role())
    print("get_type: ", ai_message.get_type())

    print("\n")
    print("--------------------")
    print("\n")

    aimessages = AIMessages()
    aimessages.add_message(AIMessage(message="Hello", sender='user', role="user", class_type='MessageAI'))
    aimessages.add_message(AIMessage(message="Hi", sender="assistant", role="assistant", class_type='MessageAI'))
    aimessages.add_message(AIMessage(message="How are you?", sender="user", role="user", class_type='MessageAI'))
    aimessages.add_message(ai_message)

    print("get_messages_formatted: ", aimessages.prints_messages_role())

    file_path = "test_messages.json"
    aimessages.write_messages_to_json(file_path)
    print(f"written to {file_path}")

    print(
        "read from JSON: ",
        AIMessages.read_messages_from_json(file_path).prints_messages_role(),
    )

    print("\n")
    print("--------------------")
    print("\n")

    struct_message = {
        "message": "Hello",
        "Trust": "3",
        "Time": "11 AM",
        "Location": "Graz",
        "Emotion": "Neutral",
    }

    serialised_message = json.dumps(struct_message)
    m = AIMessage(serialised_message, "assistant")
    messages_dict = m.print_message_role()
    print("serialised_message: ", messages_dict)

    print("\n")
    print("--------------------")
    print("\n")

    aimessages = AIMessages()
    aimessage = AIMessage("Hello", "user")
    aimessages.add_message(aimessage)
    aimessages.add_message(AIMessage("Hi", "assistant"))
    aimessages.add_message(AIMessage("How are you?", "user"))

    # b = [i.__dict__ for i in aimessages.get_messages()]
    b = aimessages.prints_messages_role()
    print(b)
    print("--------------------")

    json_string2 = json.dumps(b)
    print(json_string2)
    print("--------------------")

    d2 = json.loads(json_string2)
    print(d2)
