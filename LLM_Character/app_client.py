import websocket
import json
import threading

from LLM_Character.communication.incoming_messages import MessageType, PromptData, PromptMessage

# Set the user_id
USER_ID = "12345"  # Replace this with the actual user_id

def on_message(ws, message):
    """Callback when a message is received from the server."""
    print(f"Received from server: {message}")

def on_error(ws, error):
    """Callback when an error occurs."""
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    """Callback when the WebSocket is closed."""
    print("Connection closed")

def on_open(ws):
    """Callback when the WebSocket connection is established."""
    print("Connection opened")

    # Start a thread to handle sending multiple messages
    def send_messages():
        while True:
            user_message = input("Enter a message to send (or type 'exit' to quit): ")
            if user_message.lower() == "exit":
                ws.close()
                break
            
            msg = PromptMessage(
                type=MessageType.PROMPTMESSAGE,
                data=PromptData(persona_name="Test", user_name=USER_ID, message=user_message),
            )
            msg_json = msg.model_dump_json()
            ws.send(msg_json)
            print("Message sent to server")

    # Run the message-sending loop in a separate thread
    threading.Thread(target=send_messages, daemon=True).start()

if __name__ == "__main__":
    # Add the user_id as a query parameter in the WebSocket URL
    #ws_url = f"ws://localhost:8765/ws?user_id={USER_ID}"
    ws_url = f"wss://ai-agents-backend.gamelabgraz.at/ws?user_id={USER_ID}"
    print('User id: ' + ws_url)

    # Create the WebSocket object
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    # Run the WebSocket client (blocking call)
    ws.run_forever()
