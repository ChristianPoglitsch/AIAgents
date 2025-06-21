import logging
import time

from LLM_Character.communication.comm_medium import CommMedium
from LLM_Character.communication.message_processor import MessageProcessor
from LLM_Character.communication.reverieserver_manager import ReverieServerManager
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.util import LOGGER_NAME, setup_logging

# NOTE: ibrahim: temporary function that will be replaced in the future
# by the hungarian team ?
# which will use grpc, for multi client - (multi server?) architecture?
# somehow a manager will be needed to link the differnt clients to the
# right available servers, which is now implemented by the ReverieServerManager

logger = logging.getLogger(LOGGER_NAME)


def start_server(
    sock: CommMedium,
    serverm: ReverieServerManager,
    dispatcher: MessageProcessor,
    model: LLM_API,
):
    logger.info("listening ...")

    # Load game
    query_introduction = '{"data":{"fork_sim_code":"FORK123","sim_code":"SIM456"},"type":"StartMessage"}'
    logger.info(f"Received some juicy data : {query_introduction}")
    value = dispatcher.validate_data(sock, str(query_introduction))
    dispatcher.dispatch(sock, serverm, model, value)

    while True:
        time.sleep(1)
        query = input("Chat: ")
        if query == "q":
            break

        query_introduction = '{"data":{"persona_name":"Camila","user_name":"Chris","message":"' + query + '"},"type":"PromptMessage"}'

        logger.info(f"Received some juicy data : {query_introduction}")
        value = dispatcher.validate_data(sock, str(query_introduction))
        if value is None:
            continue

        # NOTE: should be disptached in a seperate thread, but as python has the GIL,
        # true multithreading won't work. pub-sub mechanism will be needed.
        dispatcher.dispatch(sock, serverm, model, value)


if __name__ == "__main__":
    setup_logging("python_server_endpoint")
    import torch

    from LLM_Character.communication.udp_comms import UdpComms
    from LLM_Character.llm_comms.llm_openai import OpenAIComms
    from LLM_Character.llm_comms.llm_local import LocalComms

    logger.info("CUDA found " + str(torch.cuda.is_available()))

    model = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    #model = OpenAIComms()
    #model_id = "gpt-4"

    model.init(model_id)
    wrapped_model = LLM_API(model, debug = True)

    sock = UdpComms(
        udp_ip="127.0.0.1",
        port_tx=9090,
        port_rx=9091,
        enable_rx=True,
        suppress_warnings=True,
    )
    dispatcher = MessageProcessor()

    # FIXME for example, for each new incoming socket,
    # new process/thread that executes start_server,
    server_manager = ReverieServerManager()
    start_server(sock, server_manager, dispatcher, wrapped_model)
