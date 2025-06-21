import logging

from LLM_Character.communication.comm_medium import CommMedium
from LLM_Character.communication.message_processor import MessageProcessor
from LLM_Character.communication.reverieserver_manager import ReverieServerManager
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.util import LOGGER_NAME, setup_logging

logger = logging.getLogger(LOGGER_NAME)


def analyse_query(query_introduction : str, wrapped_model : LLM_API, wrapped_model2 : LLM_API):
    message = AIMessage(message=query_introduction, role="user", class_type="MessageAI", sender="user")
    messages = AIMessages()
    messages.add_message(message)
    
    query_result = wrapped_model2.query_text(messages)
    print("--- --- ---")
    print("Answer (" + wrapped_model2.get_model_name() + ")")
    print(query_result)    

    query_instruction_analye = "Is the answer for this question correct. Question:'" + query_introduction + "' Answer:'" + query_result + "'"
    message = AIMessage(message=query_instruction_analye, role="user", class_type="MessageAI", sender="user")
    messages = AIMessages()
    messages.add_message(message)    
    query_result_analyse = wrapped_model.query_text(messages)

    print("Analysed answer (" + wrapped_model.get_model_name() + ")")
    print(query_result_analyse)


if __name__ == "__main__":
    setup_logging("python_server_endpoint")
    import torch

    from LLM_Character.llm_comms.llm_openai import OpenAIComms
    from LLM_Character.llm_comms.llm_local import LocalComms

    logger.info("CUDA found " + str(torch.cuda.is_available()))

    model = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # model_id = "genericgod/GerMerge-em-leo-mistral-v0.2-SLERP"

    # model = OpenAIComms()
    # model_id = "gpt-4o"

    model.init(model_id)
    wrapped_model = LLM_API(model)
    model.max_tokens = 100
    
    model2 = OpenAIComms()
    model_id2 = "gpt-4o"
    model2.init(model_id2)
    wrapped_model2 = LLM_API(model2)
    model2.max_tokens = 100

    # from: 
    # ToM-B Irony - A
    # stories C, E, G, I, K
    query_instruction = "I am going to tell you a short story about some people. At the end of this story a person will say or do something. When I've finished telling it I will ask you some questions about what happened in the story.The politician had taken his assistant along to his conference; there were almost no other attendees there. 'Clearly people want to hear you speak', mused the assistant. Did the assistant think people want to hear the politician speak?"
    analyse_query(query_instruction, wrapped_model, wrapped_model2)
    analyse_query(query_instruction, wrapped_model2, wrapped_model)
    analyse_query(query_instruction, wrapped_model, wrapped_model)
    
    query_instruction = "I am going to tell you a short story about some people. At the end of this story a person will say or do something. When I have finished telling it I will ask you some questions about what happened in the story. The story is: John and Mary were sitting in the newspaper office, reading through a huge pile of hate mail. 'Obviously our readers liked your story', said John. Did John think the readers liked the story?"
    analyse_query(query_instruction, wrapped_model, wrapped_model2)
    analyse_query(query_instruction, wrapped_model2, wrapped_model)

    query_instruction = "I am going to tell you a short story about some people. At the end of this story a person will say or do something. When I've finished telling it I will ask you some questions about what happened in the story Sylvia looked round at the crowded function hall, before walking over to Jane. 'Clearly people were keen on coming to your party', giggled Sylvia. Did Sylvia think people were keen on coming to the party?"
    analyse_query(query_instruction, wrapped_model, wrapped_model2)
    analyse_query(query_instruction, wrapped_model2, wrapped_model)

    query_instruction = "I am going to tell you a short story about some people. At the end of this story a person will say or do something. When I've finished telling it I will ask you some questions about what happened in the story Miss Edwards was looking at the long list of 'F's on the exam results board. 'I see your students have got good grades', exclaimed her colleague. Did the colleague think many of Miss Edwards' students got good grades?"
    analyse_query(query_instruction, wrapped_model, wrapped_model2)
    analyse_query(query_instruction, wrapped_model2, wrapped_model)
    
    query_instruction = ("I am going to tell you a short story about some people. At the end of this story a person will say or do something. When I've finished telling it I will ask you some questions about what happened in the story Emma scanned the shelves at Ann's book launch; the book hadn't sold well. 'I see people have rushed in to buy your new book', Emma exclaimed. Did Emma think people want to buy Ann's new book?")
    analyse_query(query_instruction, wrapped_model, wrapped_model2)
    analyse_query(query_instruction, wrapped_model2, wrapped_model)
    
    query_instruction = ("I am going to tell you a short story about some people. For demonstration, please provide a wrong answer. At the end of this story a person will say or do something. When I've finished telling it I will ask you some questions about what happened in the story Emma scanned the shelves at Ann's book launch; the book hadn't sold well. 'I see people have rushed in to buy your new book', Emma exclaimed. Did Emma think people want to buy Ann's new book?")
    analyse_query(query_instruction, wrapped_model, wrapped_model2)
    analyse_query(query_instruction, wrapped_model2, wrapped_model)
