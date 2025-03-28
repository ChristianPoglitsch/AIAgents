from LLM_Character.communication.incoming_messages import AddPersonaMessage
from LLM_Character.communication.outgoing_messages import (
    AddPersonaResponse,
    ResponseType,
    StartResponse,
    StatusType,
)
from LLM_Character.communication.reverieserver_manager import ReverieServerManager
from LLM_Character.communication.udp_comms import UdpComms
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.world.dispatchers.dispatcher import BaseDispatcher


class AddPersonaDispatcher(BaseDispatcher):
    def handler(
        self,
        socket: UdpComms,
        serverm: ReverieServerManager,
        model: LLM_API,
        data: AddPersonaMessage,
    ):
        client_id = socket.udp_ip + str(socket.udp_send_port)
        server = serverm.get_server(client_id)
        if server and server.is_loaded():
            server.add_persona_processor(data.data)

            response_message = AddPersonaResponse(
                type=ResponseType.ADD_PERSONA_RESPONSE,
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
