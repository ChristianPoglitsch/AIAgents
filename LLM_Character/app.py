from asyncio.windows_events import NULL
from flask import Flask, request
import json
from waitress import serve

from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.communication.incoming_messages import InitAvatar, MessageType, PromptMessage
from LLM_Character.communication.incoming_messages import PromptMessage
from LLM_Character.communication.outgoing_messages import (
    PromptResponseData,
)
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from flask import Flask
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)

from abc import ABC, abstractmethod
from copy import deepcopy

max_token_quick_reply = 25

class MessageStruct():

    def __init__(self, instruction : AIMessage):
        self._instruction = instruction
        self._chat_history = AIMessages()
        self._emotion = ' '
        self._conversation_end = False
        self._query_result = ' '
        
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

    def set_emotion(self, emotion : str):
        self._emotion = emotion

    def get_emotion(self) -> str:
        return self._emotion

    def set_query_result(self, query_result : str):
        self._query_result = query_result

    def get_query_result(self) -> str:
        return self._query_result

    def set_conversation_end(self, end : str):
        self._conversation_end = end

    def get_conversation_end(self) -> str:
        return self._conversation_end
        
class MessageProcessing(ABC):
    
    def __init__(self, decorator):
        self._decorator = decorator

    @abstractmethod
    def get_messages(self, query : str) -> MessageStruct:
        pass

class BaseDecorator(MessageProcessing):
    def __init__(self, messageStruct : MessageStruct):
        self._message_struct = messageStruct
    
    def get_messages(self, query : str) -> MessageStruct:
        return deepcopy(self._message_struct)

class EmotionDecorator(MessageProcessing):
    
    def __init__(self, decorator : MessageProcessing, llm_api : LLM_API):
        super(EmotionDecorator, self).__init__(decorator)      
        self._llm_api = llm_api

    def get_messages(self, query : str) -> MessageStruct:
        message_processing = self._decorator.get_messages(query)
        messages = message_processing.get_history().get_messages()
        
        message = 'Based on the Instruction and the chat history estimate the emotional state of the agent with one of these emotions: happy, angry, disgust, fear, surprise, sad or neutral. Answer only with the emotion.\n'
        message = message + 'Instruction: ' + message_processing.get_instruction().get_message() + '\n'    
        message = message + 'Message: ' + messages[-1].message + '\n'        
        query = AIMessage(message=message, role="user", class_type="MessageAI", sender="user")
        queries = AIMessages()
        queries.add_message(query)
        
        self._llm_api.set_max_tokens(max_token_quick_reply)
        query_result = self._llm_api.query_text(queries)
        m = message_processing.get_instruction()
        m.message = m.message + '\nYour current emotion: ' + query_result
        message_processing.set_instruction(m)
        message_processing.set_emotion(query_result)

        self._llm_api.set_max_tokens(100)
        return message_processing


class ChatCompletationDecorator(MessageProcessing):
    
    def __init__(self, decorator : MessageProcessing, llm_api : LLM_API):
        super(ChatCompletationDecorator, self).__init__(decorator)      
        self._llm_api = llm_api

    def get_messages(self, query : str) -> MessageStruct:
        message_processing = self._decorator.get_messages(query)
        query_result = self._llm_api.query_text(message_processing.get_instruction_and_history())
        message = AIMessage(message=query_result, role="assistant", class_type="MessageAI", sender="assistant")
        message_processing.add_message(message)
        message_processing.set_query_result(query_result)
        return message_processing

class ChatOverDecorator(MessageProcessing):
    
    def __init__(self, decorator : MessageProcessing, llm_api : LLM_API):
        super(ChatOverDecorator, self).__init__(decorator)      
        self._llm_api = llm_api

    def get_messages(self, query : str) -> MessageStruct:
        message_processing = self._decorator.get_messages(query)
        instruction = message_processing.get_instruction_and_history()
        message = AIMessage(message='This is the agent instruction and the chat history: ' + instruction.prints_messages_role() + ' Estimate if the conversation is over. The conversation is over if the goal of the conversation is reached, if someone says good bye or if the secret information is discovered. Reply with 1 for true and 0 for false. Only reply the number.', role="assistant", class_type="MessageAI", sender="assistant")     
        queries = AIMessages()
        queries.add_message(message)
        query_result = self._llm_api.query_text(queries)
        message_processing.set_conversation_end(query_result)
        return message_processing

def init_session(background : str, mood : str, conversation_goal : str, user_id : str):
    print(background)

    wrapped_model = get_model_openai()
    message = 'Based on the background story create additional content for a role play agent: ' + background + ' You can put here secrect information like hidden information for a game or about your personality. Only the game master can see this information. Only reply new information so that it can be added to the background story.'  
    query = AIMessage(message=message, role="user", class_type="MessageAI", sender="user")
    queries = AIMessages()
    queries.add_message(query)
    secret_information = wrapped_model.query_text(queries)

    #print(secret_information)
    message = AIMessage(message='We are playing a role game. Stay in the role. Be creative about your role. Try not repeat text. Keep your answers short. The role is: ' + background + ' This is the initial emotion: ' + mood + ' This is the goal of the conversation: ' + conversation_goal + ' This is th secret information created for you: ' + secret_information, role="user", class_type="Introduction", sender="user")
    print(message.get_message())
    message_manager = MessageStruct(message)
    message = AIMessage(message='hi', role="assistant", class_type="MessageAI", sender="assistant")
    message_manager.add_message(message)
    messages_dict[user_id] = message_manager        

def get_model_openai() -> LLM_API:
    model = OpenAIComms()
    model_id = "gpt-4o"
    model.max_tokens = 2048
    model.init(model_id)
    wrapped_model = LLM_API(model)
    return wrapped_model

def create_decorator(message_struct : MessageStruct, model : LLM_API) -> MessageProcessing:
    message_decorator = BaseDecorator(message_struct)
    message_decorator = EmotionDecorator(message_decorator, model)
    message_decorator = ChatCompletationDecorator(message_decorator, model)
    message_decorator = ChatOverDecorator(message_decorator, model)
    return message_decorator

def process_message(query : AIMessage, user_id : str):
    
    print('User id: ' + user_id)
    if not user_id in messages_dict:
        response_data = PromptResponseData(
            utt='User not found', emotion=' ', trust_level=str(0), end=bool(query_result_end)
        )    
        sending_str = response_data.model_dump_json()
        return sending_str
        
    print(query)
    wrapped_model = get_model_openai()   

    message_manager = messages_dict[user_id]
    message = AIMessage(message=query, role="user", class_type="MessageAI", sender="user")
    message_manager.add_message(message)
    decorator = create_decorator(message_manager, wrapped_model)
    decorator_result = decorator.get_messages(query)

    messages_dict[user_id] = decorator_result
    query_result_end = False

    response_data = PromptResponseData(
        utt=decorator_result.get_query_result(), emotion=decorator_result.get_emotion(), trust_level=str(0), end=bool(int(decorator_result.get_conversation_end()))
    )
    
    sending_str = response_data.model_dump_json()    
    print(f"Sending value: {response_data}")
    
    return sending_str


@sock.route('/ws')
def websocket(ws):
    
    user_id = request.args.get('user_id')
    print(user_id)
    print(f"User ID: {user_id}")

    if not user_id:
        print("Invalid user ID: User ID is required")
        user_id = 'test'

    print(f"Start server")
    while True:
        # Receive data from the client
        data = ws.receive()
        data = json.loads(data)
        
        if(data['type']==MessageType.STARTMESSAGE.value):
            pm = InitAvatar(**data)
            init_session(pm.data.background_story, pm.data.mood, pm.data.conversation_goal, user_id)
            
        elif(data['type']==MessageType.PROMPTMESSAGE.value):
            pm = PromptMessage(**data) 
            print(f"Receiving value: {pm.data.message}")
            sending_str = process_message(pm.data.message, user_id)
            print(f"Sending value: {sending_str}")

            # Send data back to the client
            ws.send(sending_str)

@app.route('/')
def hello_world():
    return '<p>Hello, World!</p>'

def run_local_chat():
    user_id = 'Test'

    #message = AIMessage(message='You are playing a role. Answer according to your character. You are a 22-year-old woman named Ana. You are from Graz and you have a physical body. Keep your response short. ', role="user", class_type="Introduction", sender="user")
    message = AIMessage(message='Let us play who are you. Randomly select one famous real or fictional person and I have to guess it. ', role="user", class_type="Introduction", sender="user")
    init_session(message.get_message(), 'Happy', ' ', user_id)

    while True:
        message = input("Chat: ")
        if message == "q":
            break     

        _ = process_message(message, user_id)

# docker save flaskhelloworld > hello.tar
if __name__ == '__main__':
    
    messages_dict = { } #AIMessages()

    if False:
        app.run(port=8765, host='0.0.0.0')
    else:
        run_local_chat()
