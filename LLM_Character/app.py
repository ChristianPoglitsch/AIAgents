from flask import Flask
import logging
import asyncio
import json
import websockets
import torch

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
from LLM_Character.util import LOGGER_NAME, setup_logging


# Define the server logic
async def echo(websocket):
    print("Client connected.")
    async for message in websocket:
        print(f"Received value: {message}")
        data = json.loads(message)
        pm = PromptMessage(**data)
        
        message = AIMessage(message=pm.data.message, role="user", class_type="MessageAI", sender="user")
        messages.add_message(message)
        query_result = wrapped_model.query_text(messages)
        message = AIMessage(message=query_result, role="assistant", class_type="MessageAI", sender="assistant")
        messages.add_message(message)
        
        response_data = PromptResponseData(
            utt=query_result, emotion='happy', trust_level=str(0), end=False
        )
        response_message = PromptReponse(
            type=ResponseType.PROMPT_RESPONSE,
            status=StatusType.SUCCESS,
            data=response_data,
        )
        sending_str = response_message.model_dump_json()
        print(f"Sending value: {sending_str}")
        await websocket.send(sending_str)

# Run the WebSocket server
async def main():
    async with websockets.serve(echo, "localhost", 8765):
        print("WebSocket server started.")
        await asyncio.Future()  # Run forever



from flask import Flask, render_template
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)

@sock.route('/ws')
def websocket(ws):
    while True:
        # Receive data from the client
        data = ws.receive()
        data = json.loads(data)
        pm = PromptMessage(**data)

        message = AIMessage(message=pm.data.message, role="user", class_type="MessageAI", sender="user")
        messages.add_message(message)
        query_result = wrapped_model.query_text(messages)
        message = AIMessage(message=query_result, role="assistant", class_type="MessageAI", sender="assistant")
        messages.add_message(message)
        
        response_data = PromptResponseData(
            utt=query_result, emotion='happy', trust_level=str(0), end=False
        )
        response_message = PromptReponse(
            type=ResponseType.PROMPT_RESPONSE,
            status=StatusType.SUCCESS,
            data=response_data,
        )
        sending_str = response_message.model_dump_json()
        print(f"Sending value: {sending_str}")

        # Send data back to the client
        ws.send(sending_str)



# docker save flaskhelloworld > hello.tar
if __name__ == '__main__':

    setup_logging("python_server_endpoint")
    logger = logging.getLogger(LOGGER_NAME)
    
    from LLM_Character.llm_comms.llm_openai import OpenAIComms
    from LLM_Character.llm_comms.llm_local import LocalComms

    logger.info("CUDA found " + str(torch.cuda.is_available()))

    messages = AIMessages()

    #model = LocalComms()
    #model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    # model_id = "genericgod/GerMerge-em-leo-mistral-v0.2-SLERP"

    model = OpenAIComms()
    model_id = "gpt-4o"

    model.init(model_id)
    wrapped_model = LLM_API(model)    
    model.max_tokens = 4096
    
    # role
    message = AIMessage(message='You are a helpful assistant', role="user", class_type="Introduction", sender="user")
    messages.add_message(message)
    message = AIMessage(message='hi', role="assistant", class_type="MessageAI", sender="assistant")
    messages.add_message(message)
    
    #asyncio.run(main())

    from waitress import serve
    app.run(port=8765, host='0.0.0.0')
