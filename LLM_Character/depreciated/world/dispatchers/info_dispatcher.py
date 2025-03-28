from LLM_Character.communication.incoming_messages import (
    GetMetaMessage,
    GetPersonaMessage,
    GetPersonasMessage,
    GetSavedGamesMessage,
    GetUsersMessage,
)
from LLM_Character.communication.outgoing_messages import (
    GetMetaResponse,
    GetPersonaResponse,
    GetPersonasResponse,
    GetSavedGamesResponse,
    GetUsersResponse,
    ResponseType,
    StartResponse,
    StatusType,
)
from LLM_Character.communication.reverieserver_manager import ReverieServerManager
from LLM_Character.communication.udp_comms import UdpComms
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.world.dispatchers.dispatcher import BaseDispatcher


class GetMetaDataDispatcher(BaseDispatcher):
    def handler(
        self,
        socket: UdpComms,
        serverm: ReverieServerManager,
        model: LLM_API,
        data: GetMetaMessage,
    ):
        client_id = socket.udp_ip + str(socket.udp_send_port)
        server = serverm.get_server(client_id)
        if server and server.is_loaded():
            info = server.get_meta_data()
            response_message = GetMetaResponse(
                type=ResponseType.GET_META_RESPONSE,
                status=StatusType.SUCCESS,
                data=info,
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


class GetPersonaDetailsDispatcher(BaseDispatcher):
    def handler(
        self,
        socket: UdpComms,
        serverm: ReverieServerManager,
        model: LLM_API,
        data: GetPersonaMessage,
    ):
        client_id = socket.udp_ip + str(socket.udp_send_port)
        server = serverm.get_server(client_id)
        if server and server.is_loaded():
            info = server.get_persona_info(data.data)
            if info:
                response_message = GetPersonaResponse(
                    type=ResponseType.GET_PERSONA_RESPONSE,
                    status=StatusType.SUCCESS,
                    data=info,
                )
                sending_str = response_message.model_dump_json()
                socket.send_data(sending_str)
            else:
                response_message = GetPersonaResponse(
                    type=ResponseType.GET_PERSONA_RESPONSE,
                    status=StatusType.FAIL,
                    data="Persona name doesn't exist.",
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


class GetPersonasDispatcher(BaseDispatcher):
    def handler(
        self,
        socket: UdpComms,
        serverm: ReverieServerManager,
        model: LLM_API,
        data: GetPersonasMessage,
    ):
        client_id = socket.udp_ip + str(socket.udp_send_port)
        server = serverm.get_server(client_id)
        if server and server.is_loaded():
            info = server.get_personas()
            response_message = GetPersonasResponse(
                type=ResponseType.GET_PERSONAS_RESPONSE,
                status=StatusType.SUCCESS,
                data=info,
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


class GetUsersDispatcher(BaseDispatcher):
    def handler(
        self,
        socket: UdpComms,
        serverm: ReverieServerManager,
        model: LLM_API,
        data: GetSavedGamesMessage,
    ):
        client_id = socket.udp_ip + str(socket.udp_send_port)
        server = serverm.get_server(client_id)
        if server and server.is_loaded():
            info = server.get_saved_games()
            response_message = GetSavedGamesResponse(
                type=ResponseType.GET_SAVED_GAMES_RESPONSE,
                status=StatusType.SUCCESS,
                data=info,
            )
            sending_str = response_message.model_dump_json()
            socket.send_data(sending_str)
        else:
            response_message = StartResponse(
                type=ResponseType.STARTRESPONSE,
                status=StatusType.FAIL,
                data="Select a saved game first or start a new game.",
            )
            sending_str = response_message.model_dump_json()
            socket.send_data(sending_str)


class GetSavedGamesDispatcher(BaseDispatcher):
    def handler(
        self,
        socket: UdpComms,
        serverm: ReverieServerManager,
        model: LLM_API,
        data: GetUsersMessage,
    ):
        client_id = socket.udp_ip + str(socket.udp_send_port)
        server = serverm.get_server(client_id)
        if server and server.is_loaded():
            info = server.get_users()
            response_message = GetUsersResponse(
                type=ResponseType.GET_USERS_RESPONSE,
                status=StatusType.SUCCESS,
                data=info,
            )
            sending_str = response_message.model_dump_json()

            socket.send_data(sending_str)
        else:
            response_message = StartResponse(
                type=ResponseType.STARTRESPONSE,
                status=StatusType.FAIL,
                data="Select a saved game first or start a new game.",
            )
            sending_str = response_message.model_dump_json()
            socket.send_data(sending_str)
