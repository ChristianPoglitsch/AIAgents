from openai import OpenAI
from LLM_Character.util import API_KEY, LOGGER_NAME

client = OpenAI(api_key=API_KEY)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Today is such a great day!",
)

response.stream_to_file("output.mp3")