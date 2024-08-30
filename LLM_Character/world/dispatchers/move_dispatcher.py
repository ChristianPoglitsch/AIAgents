from LLM_Character.communication.outgoing_messages import MoveResponse, MoveResponseData, ResponseType, StartResponse, StatusType
from LLM_Character.communication.reverieserver_manager import ReverieServerManager
from LLM_Character.world.dispatchers.dispatcher import BaseDispatcher
from LLM_Character.communication.incoming_messages import MoveMessage  
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.communication.udp_comms import UdpComms

class MoveDispatcher(BaseDispatcher):
    def handler(self, socket:UdpComms, serverM:ReverieServerManager, model:LLM_API, data:MoveMessage):
        client_id = socket.udpIP + str(socket.udpSendPort)
        server = serverM.get_server(client_id)
        if server and server.is_loaded(): 
            movements = server.move_processor(data.data, model)
            
            move_response_data = MoveResponseData(**movements)
            move_response = MoveResponse(type=ResponseType.MOVERESPONSE, status=StatusType.SUCCESS, data=move_response_data)
            move_str = move_response.model_dump_json() 
            print("Done") 
            socket.SendData(move_str)
        else :
            print("Error: Select a saved game first or start a new game.")
            response_message = StartResponse(type=ResponseType.STARTRESPONSE, status=StatusType.FAIL, data="Select a saved game first or start a new game.")
            sending_str = response_message.model_dump_json() 
            socket.SendData(sending_str) 
