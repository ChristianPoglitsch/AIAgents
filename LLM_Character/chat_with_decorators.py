import logging

from LLM_Character.communication.comm_medium import CommMedium
from LLM_Character.communication.message_processor import MessageProcessor
from LLM_Character.communication.reverieserver_manager import ReverieServerManager
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.util import LOGGER_NAME, setup_logging

logger = logging.getLogger(LOGGER_NAME)
max_token_quick_reply = 25
# ---

from abc import ABC, abstractmethod
from copy import deepcopy
import re

class MessageStruct():

    def __init__(self, instruction : AIMessage):
        self._instruction = instruction
        self._chat_history = AIMessages()
        
    def add_message(self, message : AIMessage):
        self._chat_history.add_message(message)
        
    def get_instruction(self) -> AIMessage:
        return self._instruction
    
    def set_instruction(self, instruction : AIMessage):
        self._instruction = instruction
        
    def get_history(self) -> AIMessages:
        return self._chat_history
        
    def get_instruction_and_history(self) -> AIMessages:
        if len(self._chat_history.get_messages()) > 5:
            self._chat_history.remove_item(0)
            self._chat_history.remove_item(0)

        inctruction_with_history = AIMessages()
        inctruction_with_history.add_message(self._instruction)
        for item in self._chat_history.get_messages():
            inctruction_with_history.add_message(item)
        return inctruction_with_history
        
class MessageProcessing(ABC):
    
    def __init__(self, decorator):
        self._decorator = decorator

    @abstractmethod
    def get_messages(self) -> MessageStruct:
        pass

class BaseDecorator(MessageProcessing):
    def __init__(self, messageStruct : MessageStruct):
        self._message_struct = messageStruct
    
    def get_messages(self) -> MessageStruct:
        return deepcopy(self._message_struct)

class ImportanceDecorator(MessageProcessing):
    
    def __init__(self, decorator : MessageProcessing, llm_api : LLM_API):
        super(ImportanceDecorator, self).__init__(decorator)      
        self._llm_api = llm_api
        self._important_message = AIMessages()

    def get_messages(self) -> MessageStruct:
        message_processing = self._decorator.get_messages()
        messages = message_processing.get_history().get_messages()
        
        num_messages = 10
        if len(messages) < num_messages:
            return message_processing
        
        message = 'Based on the Instruction, the message and location extract the importance of the message. Score the importance in range of [0, 10]. First, determine the importance. Second, format your answer exactly like this: {"Importance": "<Your importance>"}. Answer only with this structure. \n'
        message = message + 'Instruction: ' + message_processing.get_instruction().get_message() + '\n'    
        message = message + 'Message: ' + messages[-num_messages].message + '\n' 
        
        ai_message = AIMessage(message=message, role="user", class_type="MessageAI", sender="user")
        ai_messages = AIMessages()
        ai_messages.add_message(ai_message)        
        self._llm_api.set_max_tokens(max_token_quick_reply)
        reply = self._llm_api.query_text(ai_messages)
        
        importance = re.findall(r'\d+', reply)
        importance = int(importance[-1])        
        if importance > 6:
            self._important_message.add_message(messages[-num_messages])
            self._important_message.add_message(messages[-num_messages+1])            

        print('--- ---')
        print('Importance: ' + reply)
        print('--- ---')
        
        m = message_processing.get_instruction()
        for item in self._important_message.get_messages():
            m.message = m.message + '\nImportant message of the past:' + item.message + ' from ' + item.sender
        message_processing.set_instruction(m)
        
        return message_processing

class EmotionDecorator(MessageProcessing):
    
    def __init__(self, decorator : MessageProcessing, llm_api : LLM_API):
        super(EmotionDecorator, self).__init__(decorator)      
        self._llm_api = llm_api

    def get_messages(self) -> MessageStruct:
        message_processing = self._decorator.get_messages()        
        messages = message_processing.get_history().get_messages()
        
        message = 'Based on the Instruction, the message and location extract your current emotion. Estimate the emotional state with one of these emotions: happy, angry, disgust, fear, surprise, sad or neutral. First, determine your current emotion. Second, format your answer exactly like this: {"Emotion": "<Your emotion>"}. Answer only with this structure. \n'
        message = message + 'Instruction: ' + message_processing.get_instruction().get_message() + '\n'    
        message = message + 'Message: ' + messages[-1].message + '\n' 
        
        ai_message = AIMessage(message=message, role="user", class_type="MessageAI", sender="user")
        ai_messages = AIMessages()
        ai_messages.add_message(ai_message)
        
        self._llm_api.set_max_tokens(max_token_quick_reply)
        reply = self._llm_api.query_text(ai_messages)
        print('--- ---')
        print('Emotion: ' + reply)
        print('--- ---')
        m = message_processing.get_instruction()
        m.message = m.message + '\nHow you feel:' + reply
        message_processing.set_instruction(m)

        self._llm_api.set_max_tokens(100)
        return message_processing

class ActionDecorator(MessageProcessing):
    
    def __init__(self, decorator : MessageProcessing, llm_api : LLM_API, plan : AIMessage):
        super(ActionDecorator, self).__init__(decorator)      
        self._llm_api = llm_api
        self._plan = plan

    def get_messages(self) -> MessageStruct:
        message_processing = self._decorator.get_messages()        
        messages = message_processing.get_history().get_messages()
        
        message = 'Based on the Instruction, the message and action extract your your next action(s). First, identify the current action(s). If an action is resolved, remove it and create a new action from the conversation, otherwise keep the action. Second, format your answer exactly like this: {"Action(s)": "<Your action(s)>"}. Answer only with this structure.\n'
        message = message + 'Instruction: ' + message_processing.get_instruction().get_message() + '\n'    
        message = message + 'Message: ' + messages[-1].message + '\n'
        message = message + 'Action: ' + self._plan.get_message()
        
        ai_message = AIMessage(message=message, role="user", class_type="MessageAI", sender="user")
        ai_messages = AIMessages()
        ai_messages.add_message(ai_message)
        
        self._llm_api.set_max_tokens(max_token_quick_reply)
        plan = self._llm_api.query_text(ai_messages)
        print('--- ---')
        print('Plan: ' + plan)
        print('--- ---')
        m = message_processing.get_instruction()
        m.message = m.message + '\nYour plans for the future:' + plan
        message_processing.set_instruction(m)
        self._plan.message = plan

        self._llm_api.set_max_tokens(100)
        return message_processing

class SpatialDecorator(MessageProcessing):
    
    def __init__(self, decorator : MessageProcessing, llm_api : LLM_API, locations : AIMessage, timeDate : AIMessage):
        super(SpatialDecorator, self).__init__(decorator)      
        self._llm_api = llm_api
        self._locations = locations
        self._timeDate = timeDate

    def get_messages(self) -> MessageStruct:
        message_processing = self._decorator.get_messages()        
        messages = message_processing.get_history().get_messages()
        
        message = 'Based on the Instruction, the message and location extract your current location. First, determine the current location of the conversation. Second, format your answer exactly like this: {"Location": "<Your location>"}. Answer only with this structure. \n'
        message = message + 'Instruction: ' + message_processing.get_instruction().get_message() + '\n'    
        message = message + 'Message: ' + messages[-1].message + '\n'
        message = message + 'Location: ' + self._locations.get_message() + ' Time Date: ' + self._timeDate.get_message()
        
        ai_message = AIMessage(message=message, role="user", class_type="MessageAI", sender="user")
        ai_messages = AIMessages()
        ai_messages.add_message(ai_message)
        
        self._llm_api.set_max_tokens(max_token_quick_reply)
        location = self._llm_api.query_text(ai_messages)
        print('--- ---')
        print('Current location: ' + location)
        print('--- ---')
        m = message_processing.get_instruction()
        m.message = m.message + '\nYour location for the future:' + location
        message_processing.set_instruction(m)

        self._llm_api.set_max_tokens(100)
        return message_processing

# ---


if __name__ == "__main__":
    setup_logging("python_server_endpoint")
    import torch

    from LLM_Character.llm_comms.llm_openai import OpenAIComms
    from LLM_Character.llm_comms.llm_local import LocalComms

    logger.info("CUDA found " + str(torch.cuda.is_available()))

    # AI role
    message = AIMessage(message='You are playing a role. Answer according to your character. You are a 22-year-old woman named Ana. You are from Graz and you have a physical body. Follow your action plan. Keep your response short. ', role="user", class_type="Introduction", sender="user")
    message_manager = MessageStruct(message)    
    message = AIMessage(message='hi', role="assistant", class_type="MessageAI", sender="assistant")
    message_manager.add_message(message)
    
    message_decorator = BaseDecorator(message_manager)    

    model = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    # model_id = "genericgod/GerMerge-em-leo-mistral-v0.2-SLERP"

    #model = OpenAIComms()
    #model_id = "gpt-4o"

    model.init(model_id)
    wrapped_model = LLM_API(model)    
    model.max_tokens = 100


    time_info = AIMessage(message='Current time: 12:59, Date: 2nd October 2024.', role="assistant", class_type="MessageAI", sender="assistant")
    spatial_info = AIMessage(message='Locations: Your home in Graz Austria, Cafe in Graz Austria, hiking trail near Graz Austria. Your current location is: Cafe.', role="assistant", class_type="MessageAI", sender="assistant")
    spatial_decorator = SpatialDecorator(message_decorator, wrapped_model, spatial_info, time_info)    
    plan = AIMessage(message='Find out the name of your conversation partner. Make plans to go for a hiking trip. ', role="assistant", class_type="MessageAI", sender="assistant")
    action_decorator = ActionDecorator(spatial_decorator, wrapped_model, plan)
    importance_decorator = ImportanceDecorator(action_decorator, wrapped_model)
    emotion_decorator = EmotionDecorator(importance_decorator, wrapped_model)

    while True:
        query_introduction = input("Chat: ")
        if query_introduction == "q":
            break        
        
        message = AIMessage(message=query_introduction, role="user", class_type="MessageAI", sender="user")
        message_manager.add_message(message)
        query = emotion_decorator.get_messages().get_instruction_and_history()
        
        print('--- ---')
        print(query.prints_messages_role())
        print('--- ---')
        
        query_result = wrapped_model.query_text(query)
        print(query_result)
        message = AIMessage(message=query_result, role="assistant", class_type="MessageAI", sender="assistant")
        message_manager.add_message(message)

