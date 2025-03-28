from LLM_Character.communication.incoming_messages import StartMessage
from LLM_Character.communication.outgoing_messages import (
    ResponseType,
    StartResponse,
    StatusType,
)
from LLM_Character.communication.reverieserver_manager import ReverieServerManager
from LLM_Character.communication.udp_comms import UdpComms
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.world.dispatchers.dispatcher import BaseDispatcher
from LLM_Character.world.game import ReverieServer


class StartDispatcher(BaseDispatcher):
    def handler(
        self,
        socket: UdpComms,
        serverm: ReverieServerManager,
        model: LLM_API,
        data: StartMessage,
    ):
        clientid = socket.udp_ip + str(socket.udp_send_port)

        sd = data.data
        if serverm.get_server(clientid):
            response_message = StartResponse(
                type=ResponseType.START_RESPONSE,
                status=StatusType.FAIL,
                data="Client is already connected and the game is already loaded",
            )
            sending_str = response_message.model_dump_json()
            socket.send_data(sending_str)
            return None

        server = ReverieServer(sd.sim_code, clientid, sd.fork_sim_code)
        serverm.add_connection(clientid, server)
        server.start_processor()

        response_message = StartResponse(
            type=ResponseType.START_RESPONSE,
            status=StatusType.SUCCESS,
            data="SuccessFully started the game",
        )
        sending_str = response_message.model_dump_json()
        socket.send_data(sending_str)
