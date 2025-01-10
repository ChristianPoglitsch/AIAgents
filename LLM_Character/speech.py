import os

import azure.cognitiveservices.speech as speechsdk


def speechvoice(model):
    print("Start SpeechRecognizer")
    language = "en-US"  # 'de-AT' 'en-US'
    speechConfig = speechsdk.SpeechConfig(
        subscription=os.getenv("AZURE_KEY"),
        region="westeurope",
        speech_recognition_language=language,
    )

    # ' 'de-AT-IngridNeural' 'en-US-JennyNeural'
    speechConfig.speech_synthesis_voice_name = "en-US-JennyNeural"
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speechConfig, audio_config=audio_config
    )
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speechConfig, audio_config=audio_config
    )

    speech_recognized = "Hallo"

    print("Speak into your microphone.")
    speech_recognized = speech_recognizer.recognize_once()
    print(speech_recognized.text)
    # speech_synthesis_result =
    # speech_synthesizer.speak_text_async(speech_recognized.text).get()
    # #result.text
    print("Stop SpeechRecognizer")

    response = speech(model, speech_recognized.text)
    speech_synthesis_result = speech_synthesizer.speak_text_async(response).get()

    print(response)
    print("Test finished")
    return response, speech_synthesis_result


def speech(model, message):
    print("Start LLM processing")

    chat_history = ""
    system_template = "Your role is to be an empathic agent. Your name is Camila. Get information about the user like name, age, gender and his or her live in general. "
    # many models use triple hash '###' for keywords, Vicunas are simpler:
    prompt_template = "USER: {0}\nASSISTANT:"
    temperature = 0.7
    # prompt_template (str | None, default: None ) – Template for the prompts
    # with {0} being replaced by the user message.
    with model.chat_session(system_template, prompt_template):
        response = model.generate(prompt=message, temp=temperature)

    print(response)
    print("LLM processing finished")
    return response
