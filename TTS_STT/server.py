from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import os
from flask import Flask, request
from flask import Flask
from flask_sock import Sock

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


# base path = folder where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)
sock = Sock(app)


@sock.route('/ws')
def websocket(ws):
    
    data = ws.receive()
    print(data)


# docker save flaskhelloworld > hello.tar
if __name__ == '__main__':   

    app.run(port=8765, host='0.0.0.0')

