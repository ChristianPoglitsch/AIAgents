from LLM_Character.communication.incoming_messages import (
    UpdateMetaMessage,
    UpdatePersonaMessage,
    UpdateUserMessage,
)
from LLM_Character.communication.outgoing_messages import (
    ResponseType,
    StartResponse,
    StatusType,
    UpdateMetaResponse,
    UpdatePersonaResponse,
    UpdateUserResponse,
)
from LLM_Character.communication.reverieserver_manager import ReverieServerManager
from LLM_Character.communication.udp_comms import UdpComms
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.world.dispatchers.dispatcher import BaseDispatcher


class UpdateMetaDispatcher(BaseDispatcher):
    def handler(
        self,
        socket: UdpComms,
        serverm: ReverieServerManager,
        model: LLM_API,
        data: UpdateMetaMessage,
    ):
        client_id = socket.udp_ip + str(socket.udp_send_port)
        server = serverm.get_server(client_id)
        if server and server.is_loaded():
            server.update_meta_processor(data.data)

            response_message = UpdateMetaResponse(
                type=ResponseType.UPDATE_META_RESPONSE,
                status=StatusType.SUCCESS,
                data="Successfully updated the meta data.",
            )
            sending_str = response_message.model_dump_json()
            socket.send_data(sending_str)
        else:
            response_message = StartResponse(
                type=ResponseType.START_RESPONSE,
                status=StatusType.FAIL,
                data="Select a saved game first or start a new game.",
            )
            sending_str = response_message.model_dump_json()
            socket.send_data(sending_str)


class UpdatePersonaDispatcher(BaseDispatcher):
    def handler(
        self,
        socket: UdpComms,
        serverm: ReverieServerManager,
        model: LLM_API,
        data: UpdatePersonaMessage,
    ):
        client_id = socket.udp_ip + str(socket.udp_send_port)
        server = serverm.get_server(client_id)
        if server and server.is_loaded():
            server.update_persona_processor(data.data)

            response_message = UpdatePersonaResponse(
                type=ResponseType.UPDATE_PERSONA_RESPONSE,
                status=StatusType.SUCCESS,
                data="Successfully updated the persona data.",
            )
            sending_str = response_message.model_dump_json()
            socket.send_data(sending_str)
        else:
            response_message = StartResponse(
                type=ResponseType.START_RESPONSE,
                status=StatusType.FAIL,
                data="Select a saved game first or start a new game.",
            )
            sending_str = response_message.model_dump_json()
            socket.send_data(sending_str)


class UpdateUserDispatcher(BaseDispatcher):
    def handler(
        self,
        socket: UdpComms,
        serverm: ReverieServerManager,
        model: LLM_API,
        data: UpdateUserMessage,
    ):
        client_id = socket.udp_ip + str(socket.udp_send_port)
        server = serverm.get_server(client_id)
        if server and server.is_loaded():
            server.update_user_processor(data.data)

            response_message = UpdateUserResponse(
                type=ResponseType.UPDATE_USER_RESPONSE,
                status=StatusType.SUCCESS,
                data="Successfully updated the user data.",
            )
            sending_str = response_message.model_dump_json()
            socket.send_data(sending_str)
        else:
            response_message = StartResponse(
                type=ResponseType.START_RESPONSE,
                status=StatusType.FAIL,
                data="Select a saved game first or start a new game.",
            )
            sending_str = response_message.model_dump_json()
            socket.send_data(sending_str)
