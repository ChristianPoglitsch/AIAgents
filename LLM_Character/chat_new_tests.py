import logging
from LLM_Character.chat_new import MessageStruct

from LLM_Character.communication.comm_medium import CommMedium
from LLM_Character.communication.message_processor import MessageProcessor
from LLM_Character.communication.reverieserver_manager import ReverieServerManager
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.util import LOGGER_NAME, setup_logging

logger = logging.getLogger(LOGGER_NAME)
# ---


if __name__ == "__main__":
    setup_logging("python_server_endpoint")
    import torch

    from LLM_Character.llm_comms.llm_openai import OpenAIComms
    from LLM_Character.llm_comms.llm_local import LocalComms

    logger.info("CUDA found " + str(torch.cuda.is_available()))

    # AI role
    message = AIMessage(message='You are playing a role. Answer according to your character. You are a 22-year-old woman named Ana. You are from Graz. Followe your plans. Only reveal your plans if it fits naturally into the conversation. Keep your response short. ', role="user", class_type="Introduction", sender="user")
    message_manager = MessageStruct(message)    
    message = AIMessage(message='hi', role="assistant", class_type="MessageAI", sender="assistant")
    message_manager.add_message(message)
    
    #message = 'Based on the Instruction, the Message and Location extract your current location. Answer in the format: {"Location": "<Your location>"}. Do not include any other information or text. \n'
    message = 'Based on the Instruction, the message and location extract your current location. First, determine the current location of the conversation. Second, format your answer exactly like this: {"Location": "<Your location>"}. Answer only with this structure. \n'
    message = message + 'Instruction: ' + message_manager.get_instruction().get_message() + '\n'    
    message = message + 'Message: Hello ' + '\n'
    message = message + 'Location: Cafe'  + '\nCurrent time: 12:59, Date: 2nd October 2024.'
    query = AIMessage(message=message, role="user", class_type="MessageAI", sender="assistant")
    print(query.get_message())

    messages = AIMessages()
    messages.add_message(query)
    message = AIMessage(message='hi', role="assistant", class_type="MessageAI", sender="assistant")
    messages.add_message(message)    

    model = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # model_id = "genericgod/GerMerge-em-leo-mistral-v0.2-SLERP"

    #model = OpenAIComms()
    #model_id = "gpt-4o"

    model.init(model_id)
    wrapped_model = LLM_API(model)    
    model.max_tokens = 100

    query_result = wrapped_model.query_text(messages)
    print(query_result)

