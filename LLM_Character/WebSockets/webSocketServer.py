import asyncio
import json
import string
import websockets
import struct  # For decoding binary data
from LLM_Character.communication.comm_medium import CommMedium
from LLM_Character.communication.message_processor import MessageProcessor
from LLM_Character.communication.reverieserver_manager import ReverieServerManager
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.messages_dataclass import AIMessage, AIMessages
from LLM_Character.util import LOGGER_NAME, setup_logging


# Define the server logic
async def echo(websocket):
    print("Client connected.")
    async for message in websocket:
        print(f"Received value: {message}")
        data = json.loads(message)        
        
        message = AIMessage(message=data['type'], role="user", class_type="MessageAI", sender="user")
        messages.add_message(message)
        query_result = wrapped_model.query_text(messages)

        data['type'] = query_result
        print(f"Received value: {data}")
        data = json.dumps(data)
        await websocket.send(data)

# Run the WebSocket server
async def main():
    async with websockets.serve(echo, "localhost", 8765):
        print("WebSocket server started.")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    
    from LLM_Character.llm_comms.llm_openai import OpenAIComms
    from LLM_Character.llm_comms.llm_local import LocalComms

    #logger.info("CUDA found " + str(torch.cuda.is_available()))

    messages = AIMessages()

    model = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
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

    asyncio.run(main())
