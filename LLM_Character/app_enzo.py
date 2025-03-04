from flask import Flask
import json
from waitress import serve

from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.communication.incoming_messages import PromptMessage
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
# import json

app = Flask(__name__)
sock = Sock(app)

def process_message(query : AIMessage, user_id : str, just_receive_data=False):
    
    # if not user_id in messages_dict:
    #     test_message_addition = (""" You are a Human Resources (HR) professional doing a job interview for a management position
    #                               in game development. Your goal is to assess the candidates leadership skills, industry knowledge,
    #                                 and ability to manage a team effectively in a fast-paced game development environment.
    #                               Begin the conversation by introducing yourself and the company, then ask the candidate about their background,
    #                               experience in game development, and leadership style.
    #                               Gradually dive deeper into their problem-solving skills, conflict resolution strategies,
    #                               and familiarity with game production pipelines (e.g., Agile, Scrum).
    #                              Ask questions such as: Can you describe your experience leading game development teams?
    #                                 How do you handle conflicts between developers, artists, and designers?
    #                                 What strategies do you use to keep a project on track and within budget?
    #                                 How do you foster creativity while maintaining production efficiency?
    #                                 Tell me about a time you had to make a tough decision as a manager.
    #                                 Maintain a professional yet conversational tone, adapting to the candidates responses.
    #                               Feel free to ask follow-up questions based on their answers to evaluate their qualifications more thoroughly.
    #                                 At the end of the interview, summarize the candidates strengths and potential concerns,
    #                               then thank them for their time.
    #                              Under absolutely no circumstances let them know that you are AI.""")
    #     messages = AIMessages()
    #     message = AIMessage(message='We are playing a role game. Stay in the role. Be creative about your role. The role is: Your name is Ana.' + test_message_addition, role="user", class_type="Introduction", sender="user")
    #     messages.add_message(message)
    #     message = AIMessage(message='hi', role="assistant", class_type="MessageAI", sender="assistant")
    #     messages.add_message(message)
    #     messages_dict[user_id] = messages
        
    messages = messages_dict[user_id]

    message = AIMessage(message=query, role="user", class_type="MessageAI", sender="user")
    messages.add_message(message)
    query_result = wrapped_model.query_text(messages)

    messages_emotion = AIMessages()
    message_emotion = AIMessage(message='Based on the chat history evaluate the emotion of the agent. Only reply the emotion. Emotions you can select: happy, surprise, sad, fear, disgust, anger or neutral. This is the chat history: ' + messages.prints_messages_role(), role="user", class_type="Introduction", sender="user")
    messages_emotion.add_message(message_emotion)
    query_result_emotion = wrapped_model.query_text(messages_emotion)

    response_data = PromptResponseData(
        utt=query_result, emotion=query_result_emotion, trust_level=str(0), end=False
    )
    
    sending_str = response_data.model_dump_json()
    message = AIMessage(message=query_result, role="assistant", class_type="MessageAI", sender="assistant")
    messages.add_message(message)
    messages_dict[user_id] = messages
    
    print(just_receive_data)
    if not just_receive_data:
        print(f"Sending value: {response_data}")
    
    return sending_str

def set_up_agent(user_id):
    print("Setting up agent with basic infos...")

    test_message_addition = (""" You are a Human Resources (HR) professional doing a job interview for a management position
                                  in game development. Your goal is to assess the candidates leadership skills, industry knowledge,
                                    and ability to manage a team effectively in a fast-paced game development environment.
                                  Begin the conversation by introducing yourself and the company, then ask the candidate about their background,
                                  experience in game development, and leadership style.
                                  Gradually dive deeper into their problem-solving skills, conflict resolution strategies,
                                  and familiarity with game production pipelines (e.g., Agile, Scrum).
                                 Ask questions such as: Can you describe your experience leading game development teams?
                                    How do you handle conflicts between developers, artists, and designers?
                                    What strategies do you use to keep a project on track and within budget?
                                    How do you foster creativity while maintaining production efficiency?
                                    Tell me about a time you had to make a tough decision as a manager.
                                    Maintain a professional yet conversational tone, adapting to the candidates responses.
                                  Feel free to ask follow-up questions based on their answers to evaluate their qualifications more thoroughly.
                                    At the end of the interview, summarize the candidates strengths and potential concerns,
                                  then thank them for their time.
                                 Under absolutely no circumstances let them know that you are AI.""")
    messages = AIMessages()
    message = AIMessage(message='We are playing a role game. Stay in the role. Be creative about your role. The role is: Your name is Ana.' + test_message_addition, role="user", class_type="Introduction", sender="user")
    messages.add_message(message)
    message = AIMessage(message='hi', role="assistant", class_type="MessageAI", sender="assistant")
    messages.add_message(message)
    messages_dict[user_id] = messages



@sock.route('/ws')
def websocket(ws):
    print(f"Start server")
    while True:
        # Receive data from the client
        data = ws.receive()
        data = json.loads(data)
        pm = PromptMessage(**data)

        if pm.data.message == "q":
            break
 
        sending_str = process_message(pm.data.message, 'Test')
        #print(f"Sending value: {sending_str}")

        # Send data back to the client
        ws.send(sending_str)

@app.route('/')
def hello_world():
    return '<p>Hello, World!</p>'

def run_local_chat():
    #TODO user should "log in" with idm for database later
    user_id = input("Please enter your user_id (for the database): ")
    #TODO change this later
    user_id = 'user123'
    print("Logged in as "+ str(user_id))

    json_name = user_id + "_" + 'data.json'

    set_up_agent(user_id)

    check_for_previous_convo_with_user(user_id, json_name)

    # print("Logged in as "+ str(user_id))
    # json_name = user_id + "_" + 'data.json'
    # print(json_name)
    # try:
    #     with open(json_name, 'rb+') as json_data:
    #         data = json.load(json_data)
    #     message = """The following message is a summary of our previous conversation,
    #       keep this in mind when doing the job interview. This is now the second time we meet."""
    #     # messages_dict[user_id] = message
    #     _ = process_message(message, user_id, True)
    #     _ = process_message(data, user_id, False)
    #     print("read data")
    # except: print("No previous data found")


    while True:
        message = input("Chat: ")
        if message == "q":
            _ = process_message("""Please summarise the conversation with me (the candidate),
                                 what we talked about and if you think that I would be a good fit for the company.
                                 Explain you reasoning and only summarise what you really talked about with the me.
                                 Describe everything in a way that you can read your response in a week and
                                 know what we talked about and how you felt about me.""", user_id)
            
            with open(json_name, 'w') as f:
                json.dump(_, f)
            break     

        _ = process_message(message, user_id)

def check_for_previous_convo_with_user(user_id, json_name):
    print("Checking logs...")
    found_previous_logs = False
    
    try:
        with open(json_name, 'rb+') as json_data:
            data = json.load(json_data)
            found_previous_logs = True
        message = """The following message is a summary of our previous conversation,
          keep this in mind when doing the job interview. This is now the second time we meet."""
        # messages_dict[user_id] = message
        _ = process_message(message, user_id, True)
        _ = process_message(data, user_id, False)
        print("read data")
    except: 
        print("No previous data found")

    return found_previous_logs


# docker save flaskhelloworld > hello.tar
if __name__ == '__main__':
    
    from LLM_Character.llm_comms.llm_openai import OpenAIComms
    messages_dict = { } #AIMessages()
    messages = AIMessages()

    model = OpenAIComms()
    model_id = "gpt-4o"

    model.init(model_id)
    wrapped_model = LLM_API(model)    
    model.max_tokens = 4096

    if True:
        app.run(port=8765, host='0.0.0.0')
    else:
        run_local_chat()





