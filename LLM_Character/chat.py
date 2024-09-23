import logging

from LLM_Character.communication.comm_medium import CommMedium
from LLM_Character.communication.message_processor import MessageProcessor
from LLM_Character.communication.reverieserver_manager import ReverieServerManager
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.util import LOGGER_NAME, setup_logging

logger = logging.getLogger(LOGGER_NAME)


if __name__ == "__main__":
    setup_logging("python_server_endpoint")
    import torch

    from LLM_Character.llm_comms.llm_openai import OpenAIComms
    from LLM_Character.llm_comms.llm_local import LocalComms

    logger.info("CUDA found " + str(torch.cuda.is_available()))

    messages = AIMessages()

    model = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # model_id = "genericgod/GerMerge-em-leo-mistral-v0.2-SLERP"

    # model = OpenAIComms()
    # model_id = "gpt-4o"

    model.init(model_id)
    wrapped_model = LLM_API(model)    
    model.max_tokens = 100
    
    while True:
        query_introduction = input("Chat: ")
        if query_introduction == "q":
            break        
        
        message = AIMessage(message=query_introduction, role="user", class_type="MessageAI", sender="user")
        messages.add_message(message)
        query_result = wrapped_model.query_text(messages)
        print(query_result)
        message = AIMessage(message=query_result, role="assistant", class_type="MessageAI", sender="assistant")
        messages.add_message(message)
