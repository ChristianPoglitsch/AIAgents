import csv
from datetime import datetime
import time
import os
from flask import Flask, request
import json
from openai import OpenAI
from waitress import serve
import requests

from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.communication.incoming_messages import InitAvatar, MessageType, PromptMessage
from LLM_Character.communication.incoming_messages import PromptMessage
from LLM_Character.communication.outgoing_messages import (
    PromptResponseData,
)
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.util import API_KEY, LOGGER_NAME
from flask import Flask
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)

from abc import ABC, abstractmethod
from copy import deepcopy

max_token_quick_reply = 8
max_token_chat_completion = 512
DEVELOPER = 'developer'
ASSISTENT = 'assistant'
USER = 'user'

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
        
    def get_instruction_and_history(self, n: int = -1) -> AIMessages:
        """
        Returns the instruction along with the last n messages from chat history.

        Args:
            n (int, optional): The number of most recent messages to retrieve from the chat history.
                              If -1, all messages are retrieved.

        Returns:
            AIMessages: A collection of the instruction and the last n chat history messages.
        """
        # Get all messages if n is -1, otherwise get the last n messages
        chat_messages = self._chat_history.get_messages()
        if n == -1:
            last_n_messages = chat_messages
        else:
            last_n_messages = chat_messages[-n:]

        # Initialize a new AIMessages object
        instruction_with_history = AIMessages()

        # Add the instruction as the first message
        instruction_with_history.add_message(self._instruction)

        # Add the last n messages to the new AIMessages object
        for item in last_n_messages:
            instruction_with_history.add_message(item)

        return instruction_with_history

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
        print('*** *** *** Emotion decorator')
        self._llm_api.set_max_tokens(max_token_quick_reply)

        chat_history = message_processing.get_instruction_and_history()       
        query = AIMessage(message='Based on the Instruction and the chat history estimate the emotion of the agent: happy, angry, disgust, fear, surprise, sad or neutral. Answer only with the emotion.\n', role=DEVELOPER, class_type="MessageAI", sender=DEVELOPER)
        chat_history.add_message(query)       
        
        query_result = self._llm_api.query_text(chat_history)
        m = message_processing.get_instruction()
        m.message = m.message + '\nYour current emotion: ' + query_result
        message_processing.set_instruction(m)
        message_processing.set_emotion(query_result)
        return message_processing


class MainChatDecorator(MessageProcessing):
    
    def __init__(self, decorator : MessageProcessing, llm_api : LLM_API):
        super(MainChatDecorator, self).__init__(decorator)
        self._llm_api = llm_api

    def get_messages(self, query : str) -> MessageStruct:
        message_processing = self._decorator.get_messages(query)
        print('*** *** *** Chat decorator')
        self._llm_api.set_max_tokens(max_token_chat_completion)
        query_result = self._llm_api.query_text(message_processing.get_instruction_and_history())
        message = AIMessage(message=query_result, role=ASSISTENT, class_type="MessageAI", sender=ASSISTENT)
        message_processing.add_message(message)
        message_processing.set_query_result(query_result)
        return message_processing

class ChatOverDecorator(MessageProcessing):
    
    def __init__(self, decorator : MessageProcessing, llm_api : LLM_API):
        super(ChatOverDecorator, self).__init__(decorator)
        self._llm_api = llm_api

    def get_messages(self, query : str) -> MessageStruct:        
        message_processing = self._decorator.get_messages(query)
        print('*** *** *** Chat finished decorator')
        self._llm_api.set_max_tokens(max_token_quick_reply)
        instruction = message_processing.get_instruction_and_history()
        message = AIMessage(message='Estimate if the conversation is over. The conversation is over if the goal of the conversation is reached. Reply with 1 for true and 0 for false. Only reply the number.', role=DEVELOPER, class_type="MessageAI", sender=DEVELOPER)
        instruction.add_message(message)
        query_result = self._llm_api.query_text(instruction)
        message_processing.set_conversation_end(query_result)
        return message_processing

def init_session(background : str, mood : str, conversation_goal : str, user_id : str):
    print('*** *** *** Init session ' + background)
    wrapped_model = get_model()
    
    wrapped_model.set_max_tokens(max_token_quick_reply)
    message = 'The instruction: *' + background + '* \nDoes the instruction tell to create content? Return 1 if true, else return 0. Only return the number.'
    query = AIMessage(message=message, role=DEVELOPER, class_type="MessageAI", sender=DEVELOPER)
    queries = AIMessages()
    queries.add_message(query)
    add_additional_info = wrapped_model.query_text(queries)
    add_additional_info = eval(add_additional_info)
    print(bool(add_additional_info))
    
    secret_information = ''
    wrapped_model.set_max_tokens(max_token_chat_completion)
    
    if add_additional_info:
        message = 'The background story: *' + background + '* \nThis is the goal of the game or conversation: *' + conversation_goal + '* \n Set the game state and add additional background information for the agent. Be creativ and random. Make it challenging to reach the goal of the game. '
        query = AIMessage(message=message, role=DEVELOPER, class_type="MessageAI", sender=DEVELOPER)
        queries = AIMessages()
        queries.add_message(query)
        secret_information = wrapped_model.query_text(queries)

    #print(secret_information)
    message = AIMessage(message='We are playing a role game. Stay in the role. Be creative about your role. Try not repeat text. Keep your answers short. The role is: ' + background + ' This is the initial emotion: ' + mood + ' This is the goal of the conversation: ' + conversation_goal + ' This is the secret information created for you: ' + secret_information, role=DEVELOPER, class_type="Introduction", sender=DEVELOPER)
    print(message.get_message())
    message_manager = MessageStruct(message)
    # Note: Welcome messages are not required for all models
    #message = AIMessage(message='hi', role=ASSISTENT, class_type="MessageAI", sender=ASSISTENT)
    #message_manager.add_message(message)
    messages_dict[user_id] = message_manager

def create_decorator(message_struct : MessageStruct, model : LLM_API) -> MessageProcessing:
    message_decorator = BaseDecorator(message_struct)
    message_decorator = EmotionDecorator(message_decorator, model)
    message_decorator = MainChatDecorator(message_decorator, model)
    message_decorator = ChatOverDecorator(message_decorator, model)
    return message_decorator

def write_to_csv(user_id : str, function_name: str, duration : float):
    """
    Writes data to a CSV file. Appends if the file exists; otherwise, creates it.

    Args:
        filename (str): The name of the CSV file.
        data (list): A list representing a row of data to write.
        header (list, optional): A list representing the header row. Written only if the file doesn't exist.
    """
    # Get the current directory
    current_directory = os.getcwd()  # Path to the directory where the script is executed

    # Create the file path
    filename = os.path.join(current_directory, f"{user_id}.csv")

    # Data to write to the CSV
    header = ["Function Name", "Runtime (seconds)"]

    file_exists = os.path.isfile(filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists and header:
            header = ["Time", "Function Name", "Runtime (seconds)"]
            writer.writerow(header)  # Write the header row if the file is new
        writer.writerow([timestamp, function_name, duration])  # Write the actual data row

def process_message(query : AIMessage, user_id : str):
    
    print('User id: ' + user_id)
    if not user_id in messages_dict:
        response_data = PromptResponseData(
            utt='User not found', emotion=' ', trust_level=str(0), end=bool(False), status = 0
        )    
        sending_str = response_data.model_dump_json()
        return sending_str
        
    print(query)
    wrapped_model = get_model()   

    message_manager = messages_dict[user_id]
    message = AIMessage(message=query, role=USER, class_type="MessageAI", sender=USER)
    message_manager.add_message(message)
    decorator = create_decorator(message_manager, wrapped_model)
    decorator_result = decorator.get_messages(query)

    messages_dict[user_id] = decorator_result
    

    response_data = PromptResponseData(
        utt=decorator_result.get_query_result(), emotion=decorator_result.get_emotion(), trust_level=str(0), end=bool(int(decorator_result.get_conversation_end())), status = 0
    )
    
    sending_str = response_data.model_dump_json()    
    print(f"Sending value: {response_data}")
    
    return sending_str


def get_openai_voice(character_name: str) -> str:
    voices = {
        "Camila": "nova",
        "Melissa": "shimmer",
        "Kevin": "echo",
        "Caleb": "ash",
    }
    return voices.get(character_name, "default_voice")

def process_audio(message: str, voice : str):
    # In the Future adapt audio to work for non open ai cases?
    client = OpenAI(api_key=API_KEY)
    # TODO adapt voice based on persona
    audio_response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=message,
        response_format='wav'
    )

    # audio_response.write_to_file("output.wav")
    return audio_response.response.content


def generate_image(message: str) -> str:

    client = OpenAI(api_key=API_KEY)

    response = client.images.generate(
        model="dall-e-3",
        prompt=message,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    print(response.data[0].url)

    return response.data[0].url


@sock.route('/gi')
def websocket_gi(ws):

    while True:
        # Receive data from the client
        user_id = request.args.get('user_id')
        data = ws.receive()
        
        if(data == 'Ping'):
            continue
        
        data = json.loads(data)  
        start_time = time.time()
            
        if(data['type']==MessageType.PROMPTMESSAGE.value):
            print(f"Image generation processing")
            pm = PromptMessage(**data) 
            # print(f"Receiving value: {pm.data.message}, Current Persona: {pm.data.persona_name}")
                        
            try:
                url_generated_image = generate_image(pm.data.message)
                response = requests.get(url_generated_image)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            #save_path = 'test.jpg'
            #if response.status_code == 200:
            #    with open(save_path, 'wb') as file:
            #        file.write(response.content)
            #    print(f"Image successfully downloaded to {save_path}")
            #else:
            #    print(f"Failed to retrieve the image. HTTP Status Code: {response.status_code}")

            ws.send(response.content)

        end_time = time.time()
        duration = end_time - start_time
        write_to_csv(user_id, duration, 'ImageGeneration')


@sock.route('/tts')
def websocket_tts(ws):

    while True:
        # Receive data from the client
        user_id = request.args.get('user_id')
        data = ws.receive()
        
        if(data == 'Ping'):
            continue

        data = json.loads(data)
        start_time = time.time()
            
        if(data['type']==MessageType.PROMPTMESSAGE.value):
            print(f"Text2Speech processing")
            pm = PromptMessage(**data) 
            # print(f"Receiving value: {pm.data.message}, Current Persona: {pm.data.persona_name}")
            
            try:
                ai_voice = get_openai_voice(pm.data.persona_name)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            print(f"Set voice to {ai_voice}")
            sending_audio = process_audio(pm.data.message, ai_voice)

            print(f"Sending audio... {len(sending_audio)}")
            print(f"Text2Speech processing finished")
            ws.send(sending_audio)

        end_time = time.time()
        duration = end_time - start_time
        write_to_csv(user_id, duration, 'Text2Speech')

@sock.route('/ws')
def websocket(ws):
    
    user_id = request.args.get('user_id')
    print(f"User ID: {user_id}")

    if not user_id:
        print("Invalid user ID: User ID is required")
        user_id = 'test'

    while True:
        # Receive data from the client
        data = ws.receive()
        
        if(data == 'Ping'):
            continue

        data = json.loads(data)        
        start_time = time.time()        

        if(data['type']==MessageType.STARTMESSAGE.value):
            print(f"Start server processing ")
            pm = InitAvatar(**data)
            status_message = 0
            
            try:
                init_session(pm.data.background_story, pm.data.mood, pm.data.conversation_goal, user_id)
                response_data = PromptResponseData(utt=' ', emotion=' ', trust_level=str(0), end=status_message, status = 1)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                status_message = 2
                response_data = PromptResponseData(utt=f"An unexpected error occurred: {e}", emotion=' ', trust_level=str(0), end=0, status = 1)
                
            print(f"Start server processing finished")            
            sending_str = response_data.model_dump_json() 
            ws.send(sending_str)
            
        elif(data['type']==MessageType.PROMPTMESSAGE.value):
            print(f"Start server processing ")
            pm = PromptMessage(**data) 
            print(f"Receiving value: {pm.data.message}")
            
            try:
                sending_str = process_message(pm.data.message, user_id)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                status_message = 2
                response_data = PromptResponseData(utt=f"An unexpected error occurred: {e}", emotion=' ', trust_level=str(0), end=0, status = 0)
                sending_str = response_data.model_dump_json()

            print(f"Sending value: {sending_str}")

            print(f"Start server processing finished")
            # Send data back to the client
            ws.send(sending_str)

        end_time = time.time()
        duration = end_time - start_time
        write_to_csv(user_id, duration, 'Prompt')


@app.route('/')
def hello_world():
    return '<p>Hello, World!</p>'

def run_local_chat():
    user_id = 'Test'
    try:
        generate_image('Image of a bar with people sitting in the background')
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    message = AIMessage(message='Create yourself an character with personalty, name, hobbies, interests of your choice. We are in Graz, Austria at Cafe Mild. Your are female. (Sexual or violent content is prohibited!)', role=DEVELOPER, class_type="Introduction", sender=DEVELOPER)
    #message = AIMessage(message='Let us play who are you. Randomly select one famous real or fictional person and I have to guess it. ', role=DEVELOPER, class_type="Introduction", sender=DEVELOPER)
    init_session(message.get_message(), 'Happy', ' ', user_id)

    while True:
        message = input("Chat: ")
        if message == "q":
            break     

        _ = process_message(message, user_id)

def init_model() -> LLM_API:
    model = OpenAIComms()
    model_id = "gpt-4o"
    model.max_tokens = max_token_chat_completion
    model.init(model_id)
    wrapped_model = LLM_API(model)
    return wrapped_model

def get_model() -> LLM_API:
    return wrapped_model

# docker save flaskhelloworld > hello.tar
if __name__ == '__main__':
    
    messages_dict = { } #AIMessages()
    wrapped_model = init_model()

    if True:
        app.run(port=8765, host='0.0.0.0')
    else:
        run_local_chat()
