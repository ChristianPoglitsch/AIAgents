import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Text to convert to speech
text = "Why do you make me do these examples? They're *so* generic."

# Save the speech to an audio file
output_file = "output.mp3"  # Change to "output.wav" if you prefer WAV format
engine.save_to_file(text, output_file)

# Process and save the file
engine.runAndWait()

print(f"Speech has been saved to {output_file}")