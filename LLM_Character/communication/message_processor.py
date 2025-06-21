"""
ibrahim: temporary class which will be replaced by the hungarian team?
# after all, they are going to use grpc, and so most of the socket
# programmming will dissapear no need to make it complicated for now,
# (assume single client single server), but the server (revererieserver)
# takes in client-id which makes it possible to run multiple instances for
# different clients without having a conflict.
"""

import json
import logging
from typing import Type, Union

from LLM_Character.communication.incoming_messages import (
    AddPersonaMessage,
    BaseMessage,
    GetMetaMessage,
    GetPersonaMessage,
    GetPersonasMessage,
    GetSavedGamesMessage,
    GetUsersMessage,
    MessageType,
    MoveMessage,
    PromptMessage,
    StartMessage,
    UpdateMetaMessage,
    UpdatePersonaMessage,
    UpdateUserMessage,
)
from LLM_Character.communication.outgoing_messages import (
    ErrorResponse,
    ResponseType,
    StatusType,
)
from LLM_Character.communication.udp_comms import UdpComms
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.depreciated.world.dispatchers.dispatcher import BaseDispatcher
from LLM_Character.depreciated.world.dispatchers.info_dispatcher import (
    GetMetaDataDispatcher,
    GetPersonaDetailsDispatcher,
    GetPersonasDispatcher,
    GetSavedGamesDispatcher,
    GetUsersDispatcher,
)
from LLM_Character.depreciated.world.dispatchers.move_dispatcher import MoveDispatcher
from LLM_Character.depreciated.world.dispatchers.persona_dispatcher import AddPersonaDispatcher
from LLM_Character.depreciated.world.dispatchers.prompt_dispatcher import PromptDispatcher
from LLM_Character.depreciated.world.dispatchers.start_dispatcher import StartDispatcher
from LLM_Character.depreciated.world.dispatchers.update_dispatcher import (
    UpdateMetaDispatcher,
    UpdatePersonaDispatcher,
    UpdateUserDispatcher,
)
from LLM_Character.depreciated.world.game import ReverieServer

logger = logging.getLogger()


class MessageProcessor:
    def __init__(self):
        self._dispatch_map: dict[str, BaseDispatcher] = {}
        self._validator_map: dict[str, Type[BaseMessage]] = {}

        # POST/ PUT REQUESTS
        self.register(MessageType.STARTMESSAGE.value, StartMessage, StartDispatcher())
        self.register(
            MessageType.PROMPTMESSAGE.value, PromptMessage, PromptDispatcher()
        )
        self.register(MessageType.MOVEMESSAGE.value, MoveMessage, MoveDispatcher())
        self.register(
            MessageType.UPDATE_USER_MESSAGE.value,
            UpdateUserMessage,
            UpdateUserDispatcher(),
        )
        self.register(
            MessageType.UPDATE_META_MESSAGE.value,
            UpdateMetaMessage,
            UpdateMetaDispatcher(),
        )
        self.register(
            MessageType.UPDATE_PERSONA_MESSAGE.value,
            UpdatePersonaMessage,
            UpdatePersonaDispatcher(),
        )
        self.register(
            MessageType.ADD_PERSONA_MESSAGE.value,
            AddPersonaMessage,
            AddPersonaDispatcher(),
        )

        # GET REQUESTS
        self.register(
            MessageType.GET_PERSONAS_MESSAGE.value,
            GetPersonasMessage,
            GetPersonasDispatcher(),
        )
        self.register(
            MessageType.GET_USERS_MESSAGE.value, GetUsersMessage, GetUsersDispatcher()
        )
        self.register(
            MessageType.GET_PERSONA_MESSAGE.value,
            GetPersonaMessage,
            GetPersonaDetailsDispatcher(),
        )
        self.register(
            MessageType.GET_SAVED_GAMES_MESSAGE.value,
            GetSavedGamesMessage,
            GetSavedGamesDispatcher(),
        )
        self.register(
            MessageType.GET_META_MESSAGE.value, GetMetaMessage, GetMetaDataDispatcher()
        )

    def register(
        self,
        message_type: str,
        message_class: Type[BaseMessage],
        dispatcher_class: BaseDispatcher,
    ):
        self._validator_map[message_type] = message_class
        self._dispatch_map[message_type] = dispatcher_class

    def dispatch(
        self, socket: UdpComms, server: ReverieServer, model: LLM_API, data: BaseMessage
    ):
        # FIXME: call handler in seperate process.
        # use a wroker - master architecture ...
        # maybe use indirect communication, like pub sub or
        # for example, RabbitMQ can be used with pub sub.
        self._dispatch_map[data.type.value].handler(socket, server, model, data)

    def validate_data(self, socket: UdpComms, data: str) -> Union[BaseMessage, None]:
        data: Union[BaseMessage, str] = self._validate_data(data)
        if not isinstance(data, str):
            return data

        response_message = ErrorResponse(
            type=ResponseType.ERROR_RESPONSE, status=StatusType.FAIL, data=data
        )
        sending_str = response_message.model_dump_json()
        socket.send_data(sending_str)
        return None

    def _validate_data(self, data: str) -> Union[BaseMessage, str]:
        try:
            loaded_data = json.loads(data)
            data_type = loaded_data.get("type")

            if not data_type:
                return "No 'type' field found in the data."

            schema = self._validator_map.get(data_type)
            if not schema:
                return f"No schema registered for message type: {data_type}"

            try:
                valid_data = schema(**loaded_data)
                return valid_data
            except TypeError as e:
                return f"Validation failed due to a TypeError: {e}"
            except ValueError as e:
                return f"Validation failed due to a ValueError: {e}"
            except Exception as e:
                return f"An unexpected error occurred during validation: {e}"

        except json.JSONDecodeError as e:
            return f"Failed to decode JSON: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"


if __name__ == "__main__":
    from LLM_Character.communication.incoming_messages import (
        PromptMessage,
    )
    from LLM_Character.world.dispatchers.prompt_dispatcher import PromptDispatcher

    dispatcher = MessageProcessor()

    json_string = '{"message": "hoi", "value": 1}'
    value = dispatcher.validate_data(json_string)
    assert value is None

    json_string = '{"type": "PromptMessage", "data": {"persona_name": "Camila", \
        "user_name":"Louis", "message": "hoi"}}'
    value = dispatcher.validate_data(json_string)
    assert isinstance(value, PromptMessage), f"it is somethign else {type(value)}"
