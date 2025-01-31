from flask import Flask, request
import json
from waitress import serve

from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.communication.incoming_messages import InitAvatar, MessageType, PromptMessage
from LLM_Character.communication.incoming_messages import PromptMessage
from LLM_Character.communication.outgoing_messages import (
    PromptReponse,
    PromptResponseData,
    ResponseType,
    StatusType,
)
from LLM_Character.llm_comms.llm_api import LLM_API
from flask import Flask, render_template
from flask_sock import Sock
from flask import session

app = Flask(__name__)
sock = Sock(app)

def init_session(background : str, mood : str, conversation_goal : str, user_id : str):
    print(background)    
    messages = AIMessages()
    message = AIMessage(message='We are playing a role game. Stay in the role. Be creative about your role. The role is: ' + background + ' This is the initial emotion: ' + mood + ' This is the goal of the conversation: ' + conversation_goal, role="user", class_type="Introduction", sender="user")
    messages.add_message(message)
    message = AIMessage(message='hi', role="assistant", class_type="MessageAI", sender="assistant")
    messages.add_message(message)
    messages_dict[user_id] = messages
        

def process_message(query : AIMessage, user_id : str):
    
    print('User id: ' + user_id)
    if not user_id in messages_dict:
        response_data = PromptResponseData(
            utt='User not found', emotion=query_result_emotion, trust_level=str(0), end=bool(query_result_end)
        )    
        sending_str = response_data.model_dump_json()
        return sending_str
        
    print(query)

    messages = messages_dict[user_id]

    message = AIMessage(message=query, role="user", class_type="MessageAI", sender="user")
    messages.add_message(message)
    query_result = wrapped_model.query_text(messages)

    messages_emotion = AIMessages()
    message_emotion = AIMessage(message='Based on the chat history evaluate the emotion of the agent. This is the chat history: ' + messages.prints_messages_role() + '  Emotions you can select: happy, surprise, sad, fear, disgust, anger or neutral. Only reply one emotion.', role="user", class_type="Introduction", sender="user")
    messages_emotion.add_message(message_emotion)
    query_result_emotion = wrapped_model.query_text(messages_emotion)

    messages_emotion = AIMessages()
    message_emotion = AIMessage(message='Based on the goal of the chat history evaluate if the conversation if over. Return with 1 for true and 0 for false. This is the chat history: ' + messages.prints_messages_role(), role="user", class_type="Introduction", sender="user")
    messages_emotion.add_message(message_emotion)
    query_result_end = wrapped_model.query_text(messages_emotion)

    response_data = PromptResponseData(
        utt=query_result, emotion=query_result_emotion, trust_level=str(0), end=bool(int(query_result_end))
    )
    
    sending_str = response_data.model_dump_json()
    message = AIMessage(message=query_result, role="assistant", class_type="MessageAI", sender="assistant")
    messages.add_message(message)
    #print(messages.prints_messages_role())
    messages_dict[user_id] = messages
    
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
    user_id = 'test'

    while True:
        message = input("Chat: ")
        if message == "q":
            break     

        _ = process_message(message, user_id)

# docker save flaskhelloworld > hello.tar
if __name__ == '__main__':
    
    from LLM_Character.llm_comms.llm_openai import OpenAIComms
    messages_dict = { } #AIMessages()

    model = OpenAIComms()
    model_id = "gpt-4o"

    model.init(model_id)
    wrapped_model = LLM_API(model)    
    model.max_tokens = 4096

    if True:
        app.run(port=8765, host='0.0.0.0')
    else:
        run_local_chat()
