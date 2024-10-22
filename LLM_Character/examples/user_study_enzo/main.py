# import logging
# import time

# from LLM_Character.communication.comm_medium import CommMedium
# from LLM_Character.communication.message_processor import MessageProcessor
# from LLM_Character.communication.reverieserver_manager import ReverieServerManager
# from LLM_Character.llm_comms.llm_api import LLM_API
# from LLM_Character.util import LOGGER_NAME, setup_logging

#TODO get persona files for camilla and write the setting ad her personality out
#find a way to load that data from here on so we dont have to changhe the rest of the code
# look at the other example files for this


# logger = logging.getLogger(LOGGER_NAME)


# def start_server(
#     sock: CommMedium,
#     serverm: ReverieServerManager,
#     dispatcher: MessageProcessor,
#     model: LLM_API,
# ):
#     logger.info("listening ...")
#     while True:
#         time.sleep(1)
#         byte_data = sock.read_received_data()
#         if not byte_data:
#             continue

#         logger.info(f"Received some juicy data : {byte_data}")
#         value = dispatcher.validate_data(sock, str(byte_data))
#         if value is None:
#             continue

#         # NOTE: should be disptached in a seperate thread, but as python has the GIL,
#         # true multithreading won't work. pub-sub mechanism will be needed.
#         dispatcher.dispatch(sock, serverm, model, value)


# if __name__ == "__main__":
#     setup_logging("python_server_endpoint")
#     import torch

#     from LLM_Character.communication.udp_comms import UdpComms
#     from LLM_Character.llm_comms.llm_openai import OpenAIComms
#     from LLM_Character.llm_comms.llm_local import LocalComms

#     logger.info("CUDA found " + str(torch.cuda.is_available()))

#     #TODO add openAI alternative that gets selected through parameters maybe
#     #model = OpenAIComms()
#     #model_id = "gpt-4"

#     model = LocalComms()
#     model_id = "mistralai/Mistral-7B-Instruct-v0.2"


#     model.init(model_id)
#     wrapped_model = LLM_API(model)

#     sock = UdpComms(
#         udp_ip="127.0.0.1",
#         port_tx=9090,
#         port_rx=9091,
#         enable_rx=True,
#         suppress_warnings=True,
#     )
#     dispatcher = MessageProcessor()

#     # FIXME for example, for each new incoming socket,
#     # new process/thread that executes start_server,
#     server_manager = ReverieServerManager()
#     start_server(sock, server_manager, dispatcher, wrapped_model)


import datetime
import logging

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_local import LocalComms
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.persona.cognitive_modules.converse import chatting
from LLM_Character.persona.cognitive_modules.reflect import generate_focal_points, generate_insights_and_evidence, reflect, reset_reflection_counter, run_reflect
from LLM_Character.persona.cognitive_modules.retrieve import retrieve_focal_points
from LLM_Character.persona.persona import Persona
from LLM_Character.persona.prompt_modules.converse_prompts.summarize_conversation import run_prompt_summarize_conversation
from LLM_Character.persona.prompt_modules.reflect_prompts.insight_and_guidance import run_prompt_insight_and_evidence
from LLM_Character.persona.prompt_modules.reflect_prompts.memo_convo import run_prompt_memo_convo
from LLM_Character.persona.user import User
from LLM_Character.util import BASE_DIR, LOGGER_NAME, setup_logging

if __name__ == "__main__":
    setup_logging("examples_converse")
    logger = logging.getLogger(LOGGER_NAME)

    modela = OpenAIComms()
    model_id1 = "gpt-4"
    modela.init(model_id1)

    modelb = LocalComms()
    model_id2 = "mistralai/Mistral-7B-Instruct-v0.2"
    modelb.init(model_id2)

    models = [#modela,
               modelb]
    for model in models:
        # you have to reload persona, since you cannot have one
        # chat or a part of the chat being produced by
        # one type of LLM and the other by another etc.
        # it always needs to be he same LLM.

        person = Persona("Camila")
        person.load_from_file(
            BASE_DIR + "/LLM_Character/examples/user_study_enzo/Camila"
        )
        person.scratch.curr_time = datetime.datetime(21, 3, 4)
        user = User("Louis")

        wrapped_model = LLM_API(model)
        logger.info(model.__class__.__name__)

        # -----------------------------------------------------------------------------

        message = "Hi, I'm here for the job interview!"
        response, emotion, trust, end = chatting(
            user.scratch, person.scratch, person.a_mem, message, wrapped_model
        )
        assert isinstance(response, str)
        assert emotion in [
            "neutral",
            "happy",
            "angry",
            "disgust",
            "fear",
            "surprised",
            "sad",
        ]
        assert isinstance(trust, int)
        assert 0 <= trust <= 10
        assert isinstance(end, bool)

        logger.info("data: ")
        logger.info(f"{response} | Emotion: {emotion}, Trust: {trust}, End: {end}")

        message = "I have years of experience in game development"
        response, emotion, trust, end = chatting(
            user.scratch, person.scratch, person.a_mem, message, wrapped_model
        )
        assert isinstance(response, str)
        assert emotion in [
            "neutral",
            "happy",
            "angry",
            "disgust",
            "fear",
            "surprised",
            "sad",
        ]
        assert isinstance(trust, int)
        assert 0 <= trust <= 10
        assert isinstance(end, bool)

        logger.info("data: ")
        logger.info(f"{response} | Emotion: {emotion}, Trust: {trust}, End: {end}")

        end_flag = False
        msg_count = 0
        temp_messages = ["I worked as a Lead Programmer on Street Fighter 6.", "I cant talk about this because of NDA. Bye!"]
        while not end_flag:
            message = temp_messages[msg_count]
            msg_count += 1
            response, emotion, trust, end = chatting(
                user.scratch, person.scratch, person.a_mem, message, wrapped_model
            )
            assert isinstance(response, str)
            assert emotion in [
                "neutral",
                "happy",
                "angry",
                "disgust",
                "fear",
                "surprised",
                "sad",
            ]
            assert isinstance(trust, int)
            assert 0 <= trust <= 10
            assert isinstance(end, bool)

            logger.info("data: ")
            logger.info(f"{response} | Emotion: {emotion}, Trust: {trust}, End: {end}")
            end_flag = end
        
        #LLM should do an evaluation of the participant here => prompt to give feedback about the talk
        reflect(person.scratch, person.a_mem, wrapped_model)

        # -----------------------------------------------------------------------------
        #
        user = User("Dan")

        # -----------------------------------------------------------------------------

        message = "AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
        response, emotion, trust, end = chatting(
            user.scratch, person.scratch, person.a_mem, message, wrapped_model
        )
        assert isinstance(response, str)
        assert emotion in [
            "neutral",
            "happy",
            "angry",
            "disgust",
            "fear",
            "surprised",
            "sad",
        ]
        assert isinstance(trust, int)
        assert 0 <= trust <= 10
        assert isinstance(end, bool)

        logger.info("data: ")
        logger.info(f"{response} | Emotion: {emotion}, Trust: {trust}, End: {end}")

        message = "I have no experience in anything"
        response, emotion, trust, end = chatting(
            user.scratch, person.scratch, person.a_mem, message, wrapped_model
        )
        assert isinstance(response, str)
        assert emotion in [
            "neutral",
            "happy",
            "angry",
            "disgust",
            "fear",
            "surprised",
            "sad",
        ]
        assert isinstance(trust, int)
        assert 0 <= trust <= 10
        assert isinstance(end, bool)

        logger.info("data: ")
        logger.info(f"{response} | Emotion: {emotion}, Trust: {trust}, End: {end}")

        message = "I dont know"
        response, emotion, trust, end = chatting(
            user.scratch, person.scratch, person.a_mem, message, wrapped_model
        )
        assert isinstance(response, str)
        assert emotion in [
            "neutral",
            "happy",
            "angry",
            "disgust",
            "fear",
            "surprised",
            "sad",
        ]
        assert isinstance(trust, int)
        assert 0 <= trust <= 10
        assert isinstance(end, bool)

        logger.info("data: ")
        logger.info(f"{response} | Emotion: {emotion}, Trust: {trust}, End: {end}")

        message = "AHHHHHHHHHHH I have to go bye"
        response, emotion, trust, end = chatting(
            user.scratch, person.scratch, person.a_mem, message, wrapped_model
        )
        assert isinstance(response, str)
        assert emotion in [
            "neutral",
            "happy",
            "angry",
            "disgust",
            "fear",
            "surprised",
            "sad",
        ]
        assert isinstance(trust, int)
        assert 0 <= trust <= 10
        assert isinstance(end, bool)

        logger.info("data: ")
        logger.info(f"{response} | Emotion: {emotion}, Trust: {trust}, End: {end}")
        # -----------------------------------------------------------------------------
        #


        logger.info("Start reflecting and make decision")
        # user = User("GOD")

        
        #TODO add planning?
        #TODO why does run_prompt_insight_and_evidence produce 'this is blank' so often?  => _clean_up_response in insight_and_guidance.py????
        #TODO why is run_reflect so damn slow? -> optimize?
        #start reflection
        run_reflect(person.scratch, person.a_mem, wrapped_model)
        reset_reflection_counter(person.scratch)
        
            
        #run prompt to get AI's decision
        logger.info(run_prompt_memo_convo(person.scratch, wrapped_model, "Which of the two participants (Louis and Dan) would you hire and why?"))

        #TODO mabe use this stuff too
        # logger.info(person.a_mem.get_str_seq_thoughts())
        # logger.info(person.a_mem.get_str_seq_chats())

        #TODO ADD memory modules -> LLM has to remember this convo and the decision for later evaluations

