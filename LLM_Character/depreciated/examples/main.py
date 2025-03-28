"""
The code primarily demonstrates how to use structured message types
(like StartMessage, PromptMessage, MoveMessage, etc.)
for interacting with the python server,
it demonstrates how the entry point to the python server works,
that entry point being `start_server`.

The script serves as an end-to-end example of setting up the python server,
sending various types of messages to the python server, and handling the responses.
"""

import json
import logging
import shutil
import threading

from LLM_Character.communication.incoming_messages import (
    AddPersonaMessage,
    EventData,
    FullPersonaData,
    FullPersonaScratchData,
    GetMetaMessage,
    GetPersonaMessage,
    GetPersonasMessage,
    GetSavedGamesMessage,
    GetUsersMessage,
    MessageType,
    MetaData,
    MoveMessage,
    OneLocationData,
    PerceivingData,
    PersonaData,
    PersonaScratchData,
    PersonID,
    PromptData,
    PromptMessage,
    StartData,
    StartMessage,
    UpdateMetaMessage,
    UpdatePersonaMessage,
    UpdateUserMessage,
    UserData,
)
from LLM_Character.communication.message_processor import MessageProcessor
from LLM_Character.communication.outgoing_messages import (
    AddPersonaResponse,
    ErrorResponse,
    GetMetaResponse,
    GetPersonaResponse,
    GetPersonasResponse,
    GetSavedGamesResponse,
    GetUsersResponse,
    MoveResponse,
    PromptReponse,
    StartResponse,
    UpdateMetaResponse,
    UpdatePersonaResponse,
    UpdateUserResponse,
)
from LLM_Character.communication.reverieserver_manager import ReverieServerManager
from LLM_Character.communication.udp_comms import UdpComms
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_local import LocalComms
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.main import start_server
from LLM_Character.util import BASE_DIR, LOGGER_NAME, receive, setup_logging

# --------------------------------------------------------------------------------------
# START SERVER WITH THE FOLLOWING EXAMPLES USING OPENAI.
# --------------------------------------------------------------------------------------


def running_examples(client_sock: UdpComms):
    """
    The sending and receiving operations described here 
    are normally performed by the Unity endpoint. 
    Therefore, wherever you see client_sock.receive or client_sock.send, 
    it indicates that the Unity endpoint is receiving or sending data 
    to the Python endpoint.
    """ 
    start_mess = StartMessage(
        type=MessageType.STARTMESSAGE,
        data=StartData(fork_sim_code="FORK123", sim_code="SIM456"),
    )

    json1 = start_mess.model_dump_json()
    client_sock.send_data(json1)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = StartResponse(**json.loads(response))

    # ------------------------------------------

    prompt_mess = PromptMessage(
        type=MessageType.PROMPTMESSAGE,
        data=PromptData(persona_name="Camila", user_name="Louis", message="hi"),
    )
    json2 = prompt_mess.model_dump_json()
    client_sock.send_data(json2)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = PromptReponse(**json.loads(response))

    # ------------------------------------------

    move_mess = MoveMessage(
        type=MessageType.MOVEMESSAGE,
        data=[
            PerceivingData(
                name="Camila",
                curr_loc=OneLocationData(
                    world="Graz",
                    sector="JakominiPlatz",
                    arena="Bäckerei Sorger",
                ),
                events=[
                    EventData(
                        action_event_subject="Florian",
                        action_event_predicate=None,
                        action_event_object=None,
                        action_event_description=None,
                    ),
                    # object always need to have hierarchy in subject!
                    EventData(
                        action_event_subject="Graz:Saint Martin's Church:cafe:Hammer",
                        action_event_predicate=None,
                        action_event_object=None,
                        action_event_description=None,
                    ),
                    EventData(
                        action_event_subject="Graz:Saint Martin's Church:cafe:Sword",
                        action_event_predicate=None,
                        action_event_object=None,
                        action_event_description=None,
                    ),
                    EventData(
                        action_event_subject="Graz:Saint Martin's Church:cafe:Bible",
                        action_event_predicate=None,
                        action_event_object=None,
                        action_event_description=None,
                    ),
                    EventData(
                        action_event_subject="Camila",
                        action_event_predicate="is using",
                        action_event_object="knife",
                        action_event_description="using knife to cut an arm of",
                    ),
                ],
            )
        ],
    )

    json3 = move_mess.model_dump_json()
    client_sock.send_data(json3)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = MoveResponse(**json.loads(response))
    # ------------------------------------------

    update_meta_mess = UpdateMetaMessage(
        type=MessageType.UPDATE_META_MESSAGE,
        data=MetaData(
            curr_time="July 25, 2024, 18:15:45",
            # sec_per_step=25
        ),
    )

    json4 = update_meta_mess.model_dump_json()
    client_sock.send_data(json4)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = UpdateMetaResponse(**json.loads(response))

    # -----------------------------------------

    # send move again.
    gson = move_mess.model_dump_json()
    client_sock.send_data(gson)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = MoveResponse(**json.loads(response))

    # ------------------------------------------

    update_persona_mess = UpdatePersonaMessage(
        type=MessageType.UPDATE_PERSONA_MESSAGE,
        data=PersonaData(
            name="Camila",
            scratch_data=PersonaScratchData(
                first_name="Alice",
                last_name="Smith",
                age=30,
                living_area=OneLocationData(
                    world="Graz", sector="DownTown", arena="Someone's Appartement"
                ),
                recency_w=5,
                relevance_w=7,
                importance_ele_n=3,
            ),
        ),
    )

    json5 = update_persona_mess.model_dump_json()
    client_sock.send_data(json5)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = UpdatePersonaResponse(**json.loads(response))

    # ------------------------------------------

    update_user_mess = UpdateUserMessage(
        type=MessageType.UPDATE_USER_MESSAGE,
        data=UserData(old_name="Louis", name="John Smith"),
    )

    json6 = update_user_mess.model_dump_json()
    client_sock.send_data(json6)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = UpdateUserResponse(**json.loads(response))

    # ------------------------------------------

    add_persona_mess = AddPersonaMessage(
        type=MessageType.ADD_PERSONA_MESSAGE,
        data=FullPersonaData(
            name="Ahmed",
            scratch_data=FullPersonaScratchData(
                curr_location=OneLocationData(
                    world="Graz", sector="JakominiPlatz", arena="Home"
                ),
                first_name="Ahmed",
                last_name="El Bouzid",
                age=22,
                innate="Ahmed is brash, outspoken, and unapologetically blunt.\
                    His directness often crosses the line into vulgarity, \
                    and he's known for his requent use of coarse language. \
                    Despite his harsh exterior, there’s an underlying sense \
                    of loyalty and protectiveness towards those he cares about.",
                look="Ahmed has a robust build with a thick beard \
                    and a weathered face that reflects years of hard living. \
                    His hair is dark and usually kept short. He dresses in practical,\
                    often worn clothing that reflects his no-nonsense attitude.",
                learned="Ahmed grew up in a working-class neighborhood in Casablanca,\
                    where he learned the art of street smarts and survival. \
                    His education came from life experience rather than \
                    formal schooling.\
                    He picked up his extensive vocabulary of colorful language from \
                    the rough-and-tumble social circles he navigated. His bluntness \
                    and familiarity with vulgarity are as much a product of his \
                    environment\
                    as they are a personal choice.",
                currently="Ahmed is in Casablanca, sitting in a crowded, noisy café \
                    in the heart of the city. His presence is commanding, \
                    and he’s not shy about expressing his opinions loudly and \
                    colorfully.",
                lifestyle="Ahmed’s days are characterized by a blend of laborious work \
                    and boisterous socializing. He wakes up early to start his  \
                    day at the auto repair shop, where he tackles complex mechanical \
                    problems and interacts with a diverse clientele. After a hard \
                    day's work, he heads to his favorite local café, where he meets \
                    friends and engages in animated conversations. His language is \
                    as intense as his personality; he curses freely and speaks \
                    with a raw honesty that can be off-putting to some but is \
                    appreciated \
                    by those who value his straightforwardness. Evenings are \
                    typically spent at bustling marketplaces or local watering holes,\
                    where Ahmed's laughter and loud voice contribute to the vibrant \
                    atmosphere. His lifestyle is marked by a blend of hard work and \
                    a lively social life, punctuated by his unmistakable way of \
                    speaking.",
                living_area=OneLocationData(
                    world="Graz", sector="DownTown", arena="Ahmed's Appartement"
                ),
                recency_w=5,
                relevance_w=7,
                importance_w=9,
                recency_decay=2,
                importance_trigger_max=10,
                importance_trigger_curr=6,
                importance_ele_n=3,
            ),
            spatial_data={
                "FantasyLand": {
                    "Northern Realm": {
                        "Dragon's Lair": ["Dragon", "Treasure Chest"],
                        "Ice Cavern": ["Ice Golem", "Frozen Statue"],
                    }
                }
            },
        ),
    )

    json7 = add_persona_mess.model_dump_json()
    client_sock.send_data(json7)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = AddPersonaResponse(**json.loads(response))

    # ------------------------------------------

    get_personas_mess = GetPersonasMessage(
        type=MessageType.GET_PERSONAS_MESSAGE, data=None
    )
    json8 = get_personas_mess.model_dump_json()
    client_sock.send_data(json8)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = GetPersonasResponse(**json.loads(response))

    # ------------------------------------------

    get_users_mess = GetUsersMessage(type=MessageType.GET_USERS_MESSAGE, data=None)
    json9 = get_users_mess.model_dump_json()
    client_sock.send_data(json9)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = GetUsersResponse(**json.loads(response))

    # ------------------------------------------

    get_persona_mess = GetPersonaMessage(
        type=MessageType.GET_PERSONA_MESSAGE, data=PersonID(name="Camila")
    )
    json10 = get_persona_mess.model_dump_json()
    client_sock.send_data(json10)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = GetPersonaResponse(**json.loads(response))

    # ------------------------------------------

    get_saved_games_mes = GetSavedGamesMessage(
        type=MessageType.GET_SAVED_GAMES_MESSAGE, data=None
    )

    json11 = get_saved_games_mes.model_dump_json()
    client_sock.send_data(json11)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = GetSavedGamesResponse(**json.loads(response))

    # ------------------------------------------

    get_meta_message = GetMetaMessage(type=MessageType.GET_META_MESSAGE, data=None)
    json12 = get_meta_message.model_dump_json()
    client_sock.send_data(json12)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = GetMetaResponse(**json.loads(response))

    # ------------------------------------------

    error_message = "this is an invalid json, \
    so the server should return an error message back."

    client_sock.send_data(error_message)
    response = receive(client_sock)
    logger.info(response)
    assert response is not None
    _ = ErrorResponse(**json.loads(response))


if __name__ == "__main__":
    setup_logging("examples_main")
    logger = logging.getLogger(LOGGER_NAME)

    try:
        # CLEAN UP
        shutil.rmtree(BASE_DIR + "/LLM_Character/storage/127.0.0.18080")
        shutil.rmtree(BASE_DIR + "/LLM_Character/storage/127.0.0.19090")
    except Exception:
        pass

    modela = OpenAIComms()
    modelb = LocalComms()
    model_id1 = "gpt-4"
    model_id2 = "mistralai/Mistral-7B-Instruct-v0.2"

    udp_ip = "127.0.0.1"
    port1 = 8080
    port2 = 9090
    models = [(modela, model_id1, port1)]  # , (modelb, model_id2, port2)]
    for model, model_id, port in models:
        portTX = port
        portRX = port + 1

        server_sock = UdpComms(
            udp_ip=udp_ip,
            port_tx=portTX,
            port_rx=portRX,
            enable_rx=True,
            suppress_warnings=True,
        )

        client_sock = UdpComms(
            udp_ip=udp_ip,
            port_tx=portRX,
            port_rx=portTX,
            enable_rx=True,
            suppress_warnings=True,
        )

        model.init(model_id)
        wrapper_model = LLM_API(model)

        dispatcher = MessageProcessor()
        server_manager = ReverieServerManager()

        # The python server is set up here.
        server_thread = threading.Thread(
            target=start_server,
            args=(server_sock, server_manager, dispatcher, wrapper_model),
        )
        server_thread.daemon = True
        server_thread.start()

        # Unity endpoint is going to send some messages to the python server
        running_examples(client_sock)
