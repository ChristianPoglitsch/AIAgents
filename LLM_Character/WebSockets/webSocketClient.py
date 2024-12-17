import asyncio
import websockets
import struct

async def send_and_receive():
    uri = "ws://localhost:8765"  # WebSocket server address
    
    try:
        async with websockets.connect(uri) as websocket:
            # 1. Prepare the string to send
            message = "Hello from Python Client!"

            print(f"Sending message: {message}")
            await websocket.send(message)  # Send binary data

            # 2. Wait for the response from the server
            response_data = await websocket.recv()  # Receive binary data
            print("Received raw data:", response_data)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the client
asyncio.run(send_and_receive())
