import asyncio
import string
import websockets
import struct  # For decoding binary data

# Define the server logic
async def echo(websocket):
    print("Client connected.")
    async for message in websocket:
        print(f"Received value: {message}")
        await websocket.send(message)

# Run the WebSocket server
async def main():
    async with websockets.serve(echo, "localhost", 8765):
        print("WebSocket server started.")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
